#!/usr/bin/env python3

"""Script to generate samples for reinforcement learning. (Based on svs_inference.py)"""

import os
import argparse
import logging
import shutil
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_svs.vits import VITS
from espnet2.svs.singing_tacotron.singing_tacotron import singing_tacotron
from espnet2.tasks.gan_svs import GANSVSTask
from espnet2.tasks.svs import SVSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tts.utils import DurationCalculator
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class SingingGenerate:

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None],
        model_file: Union[Path, str, None] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        use_dynamic_filter: bool = False,
        backward_window: int = 2,
        forward_window: int = 4,
        speed_control_alpha: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        vocoder_config: Union[Path, str, None] = None,
        vocoder_checkpoint: Union[Path, str, None] = None,
        discrete_token_layers: int = 1,
        mix_type: str = "frame",
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        prefer_normalized_feats: bool = False,
        svs_task: str = "svs",
        gen_wavs: bool = False,
        sample_strategy: str = "TopBottomK"
    ):
        """Initialize SingingGenerate module."""

        # setup model
        if svs_task == "svs":
            SVSTaskClass = SVSTask
        elif svs_task == "gan_svs":
            SVSTaskClass = GANSVSTask
        else:
            raise ValueError(f"Unsupported task: {svs_task}")

        model, train_args = SVSTaskClass.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.svs = model.svs
        self.normalize = model.normalize
        self.feats_extract = model.feats_extract
        self.duration_calculator = DurationCalculator()
        self.preprocess_fn = SVSTaskClass.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        self.vocoder = None
        self.prefer_normalized_feats = prefer_normalized_feats
        self.discrete_token_layers = discrete_token_layers
        self.mix_type = mix_type
        self.gen_wavs = gen_wavs
        if vocoder_checkpoint is not None:
            vocoder = SVSTaskClass.build_vocoder_from_file(
                vocoder_config, vocoder_checkpoint, model, device
            )
            if isinstance(vocoder, torch.nn.Module):
                vocoder.to(dtype=getattr(torch, dtype)).eval()
            self.vocoder = vocoder

        logging.info(f"Extractor:\n{self.feats_extract}")
        logging.info(f"Normalizer:\n{self.normalize}")
        logging.info(f"SVS:\n{self.svs}")
        if self.vocoder is not None:
            logging.info(f"Vocoder:\n{self.vocoder}")

        # setup decoding config
        decode_conf = {}
        decode_conf.update({"use_teacher_forcing": use_teacher_forcing})
        if isinstance(self.svs, VITS):
            decode_conf.update(
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
            )
        if isinstance(self.svs, singing_tacotron):
            decode_conf.update(
                threshold=threshold,
                maxlenratio=maxlenratio,
                minlenratio=minlenratio,
                use_att_constraint=use_att_constraint,
                use_dynamic_filter=use_dynamic_filter,
                forward_window=forward_window,
                backward_window=backward_window,
            )
        self.decode_conf = decode_conf

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        text: Union[torch.Tensor, np.ndarray],
        text_lengths: Union[torch.Tensor, np.ndarray, None] = None,
        singing: Union[torch.Tensor, np.ndarray, None] = None,
        singing_lengths: Union[torch.Tensor, np.ndarray, None] = None,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration_phn: Optional[torch.Tensor] = None,
        duration_phn_lengths: Optional[torch.Tensor] = None,
        duration_ruled_phn: Optional[torch.Tensor] = None,
        duration_ruled_phn_lengths: Optional[torch.Tensor] = None,
        duration_syb: Optional[torch.Tensor] = None,
        duration_syb_lengths: Optional[torch.Tensor] = None,
        slur: Optional[torch.Tensor] = None,
        slur_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        ying: Optional[torch.Tensor] = None,
        ying_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        discrete_token: Optional[torch.Tensor] = None,
        discrete_token_lengths: Optional[torch.Tensor] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # check inputs
        if self.use_sids and sids is None:
            raise RuntimeError("Missing required argument: 'sids'")
        if self.use_lids and lids is None:
            raise RuntimeError("Missing required argument: 'lids'")
        if self.use_spembs and spembs is None:
            raise RuntimeError("Missing required argument: 'spembs'")

        batch = dict(
            text=text,
            text_lengths=text_lengths,
        )

        if singing is not None:
            batch.update(singing=singing)
            batch.update(singing_lengths=singing_lengths)
        if label is not None:
            batch.update(label=label)
            batch.update(label_lengths=label_lengths)
        if midi is not None:
            batch.update(midi=midi)
            batch.update(midi_lengths=midi_lengths)
        if duration_phn is not None:
            batch.update(duration_phn=duration_phn)
            batch.update(duration_phn_lengths=duration_phn_lengths)
        if duration_ruled_phn is not None:
            batch.update(duration_ruled_phn=duration_ruled_phn)
            batch.update(duration_ruled_phn_lengths=duration_ruled_phn_lengths)
        if duration_syb is not None:
            batch.update(duration_syb=duration_syb)
            batch.update(duration_syb_lengths=duration_syb_lengths)
        if pitch is not None:
            batch.update(pitch=pitch)
            batch.update(pitch_lengths=pitch_lengths)
        if phn_cnt is not None:
            batch.update(phn_cnt=phn_cnt)
        if slur is not None:
            batch.update(slur=slur)
            batch.update(slur_lengths=slur_lengths)
        if energy is not None:
            batch.update(energy_lengths=energy_lengths)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if discrete_token is not None:
            batch.update(discrete_token=discrete_token)
            batch.update(discrete_token_lengths=discrete_token_lengths)
        batch = to_device(batch, self.device)

        with torch.no_grad():
            _, _, _, output_dict = self.model(**batch, flag_RL=True)

        logits_b = output_dict["feat_gen"]
        f0_b = output_dict["pitch"]
        feat_lengths = output_dict["feat_length"]
        bs = logits_b.size(0)
        samples_num = kwargs["samples_num"]
        output_dict["tokens_list"] = []
        output_dict["wavs_list"] = []
        for i in range(bs):
            logits = logits_b[i][: feat_lengths[i] * self.discrete_token_layers, :]
            f0 = f0_b[i][: feat_lengths[i], :]
            assert f0.size(0) * self.discrete_token_layers == logits.size(0), """
                Mismatch between logits({logits.shape}) and f0({f0.shape}) in {key}.
            """
            # sampled tokens
            if "temperature" in kwargs:
                logits = logits / kwargs["temperature"]
            token_prob = torch.softmax(logits, dim=-1)
            token_sampled = torch.multinomial(token_prob, samples_num, replacement=True)
            output_dict["tokens_list"].append(token_sampled)
            # apply vocoder (mel-to-wav)
            if self.gen_wavs:
                wavs = []
                for j in range(samples_num):
                    token_idx = token_sampled[:, j]
                    if self.vocoder is not None:
                        if self.discrete_token_layers > 1:
                            # NOTE(Yuxun): vocoder can only accept 'frame' type, [T, L]
                            if self.mix_type == "frame":
                                input_feat = token_idx.view(-1, self.discrete_token_layers)
                            elif self.mix_type == "sequence":
                                input_feat = token_idx.view(
                                    self.discrete_token_layers, -1
                                ).transpose(0, 1)

                            wav = self.vocoder(input_feat, f0.squeeze(1))
                            wavs.append(wav)
                output_dict["wavs_list"].append(wavs)
        
        output_dict.pop("feat_gen")
        output_dict.pop("pitch")

        return output_dict

    @property
    def fs(self) -> Optional[int]:
        """Return sampling rate."""
        if hasattr(self.vocoder, "fs"):
            return self.vocoder.fs
        elif hasattr(self.svs, "fs"):
            return self.svs.fs
        else:
            return None

    @property
    def use_speech(self) -> bool:
        """Return speech is needed or not in the inference."""
        return self.use_teacher_forcing or getattr(self.svs, "use_gst", False)

    @property
    def use_sids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.svs.spks is not None

    @property
    def use_lids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.svs.langs is not None

    @property
    def use_spembs(self) -> bool:
        """Return spemb is needed or not in the inference."""
        return self.svs.spk_embed_dim is not None

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        vocoder_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build SingingGenerate instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            SingingGenerate: SingingGenerate instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        if vocoder_tag is not None:
            if vocoder_tag.startswith("parallel_wavegan/"):
                try:
                    from parallel_wavegan.utils import download_pretrained_model

                except ImportError:
                    logging.error(
                        "`parallel_wavegan` is not installed. "
                        "Please install via `pip install -U parallel_wavegan`."
                    )
                    raise

                from parallel_wavegan import __version__

                # NOTE(kan-bayashi): Filelock download is supported from 0.5.2
                assert V(__version__) > V("0.5.1"), (
                    "Please install the latest parallel_wavegan "
                    "via `pip install -U parallel_wavegan`."
                )
                vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
                vocoder_file = download_pretrained_model(vocoder_tag)
                vocoder_config = Path(vocoder_file).parent / "config.yml"
                kwargs.update(
                    vocoder_config=vocoder_config, vocoder_checkpoint=vocoder_file
                )

            else:
                raise ValueError(f"{vocoder_tag} is unsupported format.")

        return SingingGenerate(**kwargs)


@typechecked
def inference(
    output_dir: Union[Path, str],
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    use_teacher_forcing: bool,
    noise_scale: float,
    noise_scale_dur: float,
    allow_variable_data_keys: bool,
    vocoder_config: Optional[str] = None,
    vocoder_checkpoint: Optional[str] = None,
    vocoder_tag: Optional[str] = None,
    discrete_token_layers: int = 1,
    mix_type: str = "frame",
    svs_task: Optional[str] = "svs",
    # rl related
    samples_num: int = 1,
    samples_dir_name: Optional[str] = "samples",
    gen_wavs: bool = False,
    temperature: float = 30.,
):
    """Perform SVS model decoding."""
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    singingGenerate = SingingGenerate(
        train_config=train_config,
        model_file=model_file,
        use_teacher_forcing=use_teacher_forcing,
        noise_scale=noise_scale,
        noise_scale_dur=noise_scale_dur,
        vocoder_config=vocoder_config,
        vocoder_checkpoint=vocoder_checkpoint,
        discrete_token_layers=discrete_token_layers,
        mix_type=mix_type,
        dtype=dtype,
        device=device,
        svs_task=svs_task,
        gen_wavs=gen_wavs,
    )

    # 3. Build data-iterator
    loader = SVSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SVSTask.build_preprocess_fn(singingGenerate.train_args, False),
        collate_fn=SVSTask.build_collate_fn(singingGenerate.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=False,
    )

    # 4. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / f"{samples_dir_name}" / "samples").mkdir(parents=True, exist_ok=True)
    sample_writer = NpyScpWriter(output_dir / f"{samples_dir_name}"/ "samples", output_dir / f"{samples_dir_name}" / "samples_idx.scp")
    sample_shape_writer = open(output_dir / f"{samples_dir_name}" / "samples_shape", "w")

    if gen_wavs:
        (output_dir / samples_dir_name / "wav").mkdir(parents=True, exist_ok=True)
        wav_writer = open(output_dir / samples_dir_name / "wav.scp", "w")

    # Lazy load to avoid the backend error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for idx, (keys, batch) in enumerate(loader, 1):
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))

        logging.info(f"keys: {keys}")

        start_time = time.perf_counter()
        output_dict = singingGenerate(**batch, samples_num=samples_num, temperature=temperature)
        logging.info(f"output_dict: {output_dict.keys()}")

        # RL data prep substage 1: get sample idx
        if output_dict.get("tokens_list"):
            for i in range(_bs):
                key = keys[i]
                logits = output_dict["tokens_list"][i]
                for j in range(samples_num):
                    sample_idx = logits[:, j]
                    uid = key + "_" + str(j)
                    sample_writer[uid] = sample_idx.cpu().numpy()
                    sample_shape_writer.write(
                        f"{uid} " + ",".join(map(str, sample_idx.shape)) + "\n"
                    )
        
        # RL data prep substage 2 [Optional]: get wavs
        if output_dict.get("wavs_list") and gen_wavs:
            for i in range(_bs):
                key = keys[i]
                wavs_sample_list = output_dict["wavs_list"][i]
                for j in range(samples_num):
                    wav = wavs_sample_list[j]
                    uid = key + "_" + str(j)
                    sf.write(
                        f"{output_dir}/{samples_dir_name}/wav/{uid}.wav",
                        wav.cpu().numpy(),
                        singingGenerate.fs,
                        "PCM_16",
                    )
                    wav_writer.write(f"{uid} {os.path.abspath(output_dir / samples_dir_name / 'wav' / f'{uid}.wav')}\n")


def get_parser():
    """Get argument parser."""

    parser = config_argparse.ArgumentParser(
        description="Generate Samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch size for samples generation",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file.",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file.",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--use_teacher_forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.667,
        help="Noise scale parameter for the flow in vits",
    )
    parser.add_argument(
        "--noise_scale_dur",
        type=float,
        default=0.8,
        help="Noise scale parameter for the stochastic duration predictor in vits",
    )

    group = parser.add_argument_group("Vocoder related")
    group.add_argument(
        "--vocoder_checkpoint",
        default="None",
        type=str_or_none,
        help="checkpoint file to be loaded.",
    )
    group.add_argument(
        "--vocoder_config",
        default=None,
        type=str_or_none,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--discrete_token_layers",
        type=int,
        default=1,
        help="layers of discrete tokens",
    )
    parser.add_argument(
        "--mix_type",
        type=str,
        default="frame",
        help="multi token mix type, 'sequence' or 'frame'.",
    )
    parser.add_argument(
        "--svs_task",
        default="svs",
        type=str_or_none,
        help="SVS task name. svs or gan_svs",
    )
    group = parser.add_argument_group("rl related")
    parser.add_argument(
        "--samples_num",
        type=int,
        default=1,
        help="number of chosen smaples (for RL)",
    )
    parser.add_argument(
        "--samples_dir_name",
        type=str,
        default=None,
        help="name of samples directory (for RL)",
    )
    parser.add_argument(
        "--gen_wavs",
        type=bool,
        default=False,
        help="whether to generate wavs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="temperature for token samples",
    )
 
    return parser


def main(cmd=None):
    """Run SVS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()

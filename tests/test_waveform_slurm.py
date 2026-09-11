"""Exercise actual launcher delegation with stubbed cluster commands."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cluster_stubs(tmp_path):
    project = tmp_path / "project with spaces"
    shutil.copytree(ROOT / "slurm", project / "slurm")
    binary = tmp_path / "bin"
    binary.mkdir()
    conda = tmp_path / "conda"
    (conda / "etc/profile.d").mkdir(parents=True)
    (conda / "etc/profile.d/conda.sh").write_text('conda() { :; }\n')
    capture = tmp_path / "capture.py"
    capture.write_text('import os, sys, json\nwith open(os.environ["CAPTURE_PATH"], "a") as f:\n'
                       ' f.write(json.dumps({"argv":sys.argv[1:],"env":dict(os.environ)})+"\\n")\n')
    stubs = {"module": ':\n', "conda": 'printf "%s\\n" "$FAKE_CONDA"\n',
             "nvidia-smi": ':\n', "scontrol": 'echo localhost\n',
             "python": 'exec "$REAL_PYTHON" "$CAPTURE_SCRIPT" "$@"\n'}
    for name, body in stubs.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    env = {"PATH": str(binary) + ":" + os.environ["PATH"], "HOME": os.environ["HOME"],
           "REAL_PYTHON": sys.executable, "CAPTURE_SCRIPT": str(capture),
           "CAPTURE_PATH": str(tmp_path / "calls.jsonl"), "FAKE_CONDA": str(conda),
           "SLURM_SUBMIT_DIR": str(project), "SLURM_JOB_ID": "123", "SLURM_JOB_NODELIST": "localhost",
           "SLURM_CPUS_PER_TASK": "2", "LOCAL_SCRATCH": "no", "TMPDIR": str(tmp_path)}
    return project, env


@pytest.mark.parametrize("script,entry", [("train_3d_waveform.sbatch", "fno_21cm_3d.py"),
                                         ("train_2d_xhi_waveform.sbatch", "fno_xhi2d.py"),
                                         ("train_zre_waveform.sbatch", "fno_zre.py")])
@pytest.mark.parametrize("overrides", [False, True])
def test_training_launchers_select_waveforms_preserve_overrides_and_delegate(cluster_stubs, script, entry, overrides):
    project, env = cluster_stubs
    env.update({"MODEL_KIND": "localfno", "LOCAL_OPERATOR": "fourier", "GLOBAL_OPERATOR": "hadamard"})
    if overrides:
        env.update({"N_EPOCHS": "3", "WAVEFORM_LOCAL_BINS": "9", "WAVEFORM_LR_RATIO": "0.2",
                    "CHECKPOINT_DIR": "checkpoints/custom run", "WAVEFORM_INIT": "sine", "WAVEFORM_TRAINING_MODE": "joint_then_kernel",
                    "WAVEFORM_ADAPT_EPOCHS": "10",
                    "WAVEFORM_PHASE_EPOCHS": "2", "WAVEFORM_KERNEL_EPOCHS": "4",
                    "WAVEFORM_FIRST_PHASE": "kernel", "WAVEFORM_KERNEL_SCOPE": "all",
                    "INIT_CHECKPOINT": "checkpoints/source run/final_model_state_dict.pt"})
    result = subprocess.run(["bash", str(project / "slurm" / script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CAPTURE_PATH"]).read_text().splitlines()]
    last = calls[-1]
    assert last["argv"] == [entry]
    settings = last["env"]
    assert settings["MODEL_KIND"] == "localop"
    assert settings["LOCAL_OPERATOR"] == settings["GLOBAL_OPERATOR"] == "learned_waveform"
    assert settings["WAVEFORM_LOCAL_BINS"] == ("9" if overrides else "15")
    assert settings["WAVEFORM_GLOBAL_BINS"] == "31"
    assert settings["WAVEFORM_INIT"] == ("sine" if overrides else "random")
    assert settings["WAVEFORM_TRAINING_MODE"] == ("joint_then_kernel" if overrides else "joint")
    assert settings["WAVEFORM_ADAPT_EPOCHS"] == ("10" if overrides else "25")
    assert settings["WAVEFORM_PHASE_EPOCHS"] == ("2" if overrides else "1")
    assert settings["WAVEFORM_KERNEL_EPOCHS"] == ("4" if overrides else "5")
    assert settings["WAVEFORM_FIRST_PHASE"] == ("kernel" if overrides else "waveform")
    assert settings["WAVEFORM_KERNEL_SCOPE"] == ("all" if overrides else "spectral")
    if overrides:
        assert settings["INIT_CHECKPOINT"] == "checkpoints/source run/final_model_state_dict.pt"
    assert settings["WAVEFORM_LR_RATIO"] == ("0.2" if overrides else "0.1")
    assert settings["N_EPOCHS"] == ("3" if overrides else "200" if entry == "fno_zre.py" else "20")
    defaults = {"fno_21cm_3d.py": "checkpoints/checkpoints_3d_lwf_lwf_plain",
                "fno_xhi2d.py": "checkpoints/checkpoints_2d_xhi_local_lwf_lwf",
                "fno_zre.py": "checkpoints/checkpoints_zre_local_lwf_lwf_l2"}
    assert settings["CHECKPOINT_DIR"] == ("checkpoints/custom run" if overrides else defaults[entry])


def test_visualization_launcher_preserves_argument_boundaries(cluster_stubs):
    project, env = cluster_stubs
    checkpoint_dir = project / "checkpoints/my run"
    checkpoint_dir.mkdir(parents=True)
    env.update({"CHECKPOINT_DIR": str(checkpoint_dir), "CHECKPOINT_KIND": "final",
                "WAVEFORM_INPUT_SHAPE": "140 140 256", "OUT_DIR": "figures/my waveforms"})
    result = subprocess.run(["bash", str(project / "slurm/viz_waveforms.sbatch")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE_PATH"]).read_text().splitlines()[-1])["argv"]
    assert args == ["-m", "viz.learned_waveforms", "--checkpoint-dir", str(checkpoint_dir),
                    "--checkpoint-kind", "final", "--max-modes", "6",
                    "--input-shape", "140", "140", "256", "--out-dir", "figures/my waveforms"]


def test_shell_syntax_and_no_trailing_sbatch_comments():
    scripts = list((ROOT / "slurm").glob("*waveform*.sbatch")) + [ROOT / "slurm/train_3d_matrix.sbatch"]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
        for line in script.read_text().splitlines():
            if line.startswith("#SBATCH"):
                assert "#" not in line[1:]

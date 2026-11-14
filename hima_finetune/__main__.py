"""hima-finetune <step> [options]   (or: python -m hima_finetune <step> [options])

  transcribe  raw recordings, no text -> clips + metadata.csv (cut at pauses, written down by Hima-ASR)
  distill     no recordings at all    -> clips + metadata.csv spoken by a big open teacher model (needs an L4/A100-class GPU)
  prepare     clips + metadata.csv    -> training data (codec latents, voice prints, a held-out set)
  train       training data           -> the adapter (resumable with --resume)
  export      the adapter             -> a complete model folder with your voices built in

Run `hima-finetune <step> -h` for the options of each step.
"""
import importlib
import sys

STEPS = ("transcribe", "distill", "prepare", "train", "export")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    if sys.argv[1] in ("-V", "--version"):
        from . import __version__; print(__version__); sys.exit(0)
    step = sys.argv[1]
    if step not in STEPS:
        sys.exit(f"unknown step '{step}': choose one of {', '.join(STEPS)}")
    sys.argv = [f"hima-finetune {step}"] + sys.argv[2:]
    importlib.import_module(f".{step}", __package__).main()


if __name__ == "__main__":
    main()

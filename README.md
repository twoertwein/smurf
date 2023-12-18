# SMURF (Statistical Modality Uniqueness and Redundancy Factorization)

The code for ["Statistical Modality Uniqueness and Redundancy Factorization"](https://TODO) at TODO.


## Setup
```sh
git clone git@github.com:twoertwein/smurf.git
cd smurf
poetry update  # installs all dependencies
```

## Usage
For SMURF (additive model)
```sh
# proposed
python train.py --dimension <dimension> --modalities=<modalities> --model=smurf
# baseline (lambda = 0)
python train.py --dimension <dimension> --modalities=<modalities> --model=base
# baseline (E-HGR)
python train.py --dimension <dimension> --modalities=<modalities> --model=ehgr
```

For MRO-SMURF (non-additive model)
```sh
# proposed
python train.py --dimension <dimension> --modalities=<modalities> --model=smurf --interactions=1
# baseline (lambda = 0, no MRO)
python train.py --dimension <dimension> --modalities=<modalities> --model=base --interactions=1
# baseline (E-HGR, no MRO)
python train.py --dimension <dimension> --modalities=<modalities> --model=ehgr --interactions=1
```

where `<dimension>` is a string specifying the dataset and task and `<modalities>` is a string specifying the set of modalities to use (to use a subset of modalities, simply specify a subset of the string, e.g., shorten `alv` to `al` to run a model using only the acoustic and language features):

- `toy/toy`: additive synthetic data (regression; modalities: `alv`)
- `natoy/toy`: non-additive synthetic data (regression; modalities: `alv`)
- `mosi/sentiment`: sentiment on MOSI  (regression; modalities: `alv`)
- `mosei/sentiment`, `mosei/happiness`: sentiment and happiness on MOSEI (regression; modalities: `alv`)
- `iemocap/valence`, `iemocap/arousal`: valence and arousal and IEMOCAP (regression; modalities: `alv`)
- `tpot/constructs`: four affective states (classification; modalities: `alv`)
- `umeme/valence`, `umeme/arousal`: valence and arousal on UMEME (regression; modalities: `alv` and `a+lv`; on the original recordings)
- `sewa/arousal`, `sewa/valence`: valence and arousal on SEWA (regression; modalities: `alv`; at the utterance-level)
- `recola/arousal`, `recola/valence`: valence and arousal on RECOLA (regression; modalities: `ahv`; at the utterance-level)
- `vreed/av`: four arousal-valence quadrants (classification; modalities: `hsv`)

## Data

The pre-processed features used for the machine-learning experiments for two synthetic tasks  and VREED are part of this git repository. The features for the remaining datasets (excluding SEWA and IEMOCAP) are available [here](https://cmu.box.com/s/u2mq4ym5qyk6cy2378b5j9yne9gzwcmp). If you want the features for [IEMOCAP](https://cmu.box.com/s/ivq6zozfbn7s2xusk74tunaafg0rllkq) and [SEWA](https://cmu.box.com/s/rzdgvtchk95mr0jyopeofob1cqprb7l5), please send us proof that you completed the data-sharing agreements required by those projects.

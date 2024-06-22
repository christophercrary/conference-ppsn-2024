# Conference – PPSN 2024

This repository was created for the conference paper "Enhancing the Computational Efficiency of Genetic Programming through Alternative Floating-Point Primitives," which was accepted to the 2024 Parallel Problem Solving from Nature (PPSN) conference.

Python-based implementations of the proposed math algorithms, along with the relevant experiments and results for measuring median relative error, are provided in `code/approximations`. Separately, a Jupyter notebook analyzing the experiments conducted by Operon is provided in `code/operon`. Files needed for running the Operon experiments and analysis script locally can be installed using the following instructions.

## Installation instructions

The following is only necessary when executing scripts locally. **Note that the relevant results can be viewed without installation by simply opening the relevant CSV or Jupyter notebook files.**

Also, when executing locally, note that only MacOS and Linux operating systems are likely supported.

### Install using Nix

This is likely to be the easiest way to run the experiment. Install [nix](https://nixos.org/download/) or [nix-portable](https://github.com/DavHau/nix-portable/releases/).

Once `nix` is installed, unpack the Operon sources and navigate to that folder. Once inside:

```bash
git init . # required by nix
git add flake.nix # required by nix
nix develop --extra-experimental-features nix-command --extra-experimental-features flakes # dev shell
```

Then, compile Operon with all the new computational backends:
```bash
for backend in {Eigen,Stl,Vdt,Mad_Transcendental_Fast,Mad_Transcendental_Faster,Mad_Transcendental_Fastest}; do
    cmake -S . -B build_${backend} --preset build-linux -DCMAKE_BUILD_TYPE=Release -DOPERON_MATH_BACKEND=${backend}
    cmake --build build_${backend} -j
done
```

Note that the cmake preset (`--preset`) must correspond to your actual system. For example, on MacOS use `--preset build-osx`.

### Install in a Conda environment

#### Prerequisites
- Ensure that some Conda package management system (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) is installed on the relevant machine.
- Download the latest software release from GitHub, available [here](https://github.com/christophercrary/conference-ppsn-2024/releases/tag/v0.1.0). Ignore the `data.zip` file for now.

Upon extracting the source code, set up the relevant Conda environment and tools by executing the following within a shell program, after having navigated to the repository directory within the shell:

```
conda env create -f environment.yml
conda activate conference-ppsn-2024
bash install.sh
```

To finish installation, extract and copy the contents of the `data.zip` file from the software release and paste them within the `code/operon` folder. These contents provide the relevant data/sources needed to run/analyze the relevant Operon experiments.

### Running an experiment

Make sure to change the shebang at the top of the `operon_experiment.py` file to the actual path to your Python executable. Assuming the Operon build folders are in the same folder with the Python script:

```bash
 for backend in {Eigen,Vdt,Stl,Mad_Transcendental_Fast,Mad_Transcendental_Faster,Mad_Transcendental_Fastest}; do ./operon_experiment.py --bin ./build_${backend}/cli/operon_nsgp --data experiment/data/ --reps 20 >> my_results.csv; done
```
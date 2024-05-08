# CONFERENCE – PPSN 2024

This repository was created for the conference paper "Enhancing the Computational Efficiency of Genetic Programming through Alternative Floating-Point Primitives," which was submitted to the 2024 Parallel Problem Solving from Nature (PPSN) conference.

Python-based implementations of the proposed math algorithms, along with the relevant experiments and results for measuring median relative error, are provided in `code/approximations`. Scripts and Jupyter notebooks regarding the analysis of experiments conducted with Operon are provided in `code/operon`. Files and information regarding the execution of the Operon experiments will be added at a later date.

## Installation instructions

The following is only necessary when executing scripts locally. **Note that the relevant results can be viewed without installation by simply opening the relevant CSV or Jupyter notebook files.** 

Also, when executing locally, note that only MacOS and Linux operating systems are likely supported.

### Prerequisites
- Ensure that some Conda package management system (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) is installed on the relevant machine.
- Download the latest software release from GitHub, available [here](https://github.com/christophercrary/conference-ppsn-2024/releases/tag/v0.1.0). Ignore the `data.zip` file for now.

Upon extracting the source code, set up the relevant Conda environment and tools by executing the following within a shell program, after having navigated to the repository directory within the shell:

```
conda env create -f environment.yml
conda activate conference-ppsn-2024
bash install.sh
```

To finish installation, extract and copy the contents of the `data.zip` file from the software release (i.e., the two folders) and paste them within the `code/operon` folder. These contents provide the data generated by the relevant Operon experiments.
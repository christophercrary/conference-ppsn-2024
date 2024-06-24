# Conference – PPSN 2024

This repository was created for the conference paper "Enhancing the Computational Efficiency of Genetic Programming through Alternative Floating-Point Primitives," which was accepted to the 2024 Parallel Problem Solving from Nature (PPSN) conference.

Python-based implementations of the proposed math algorithms, along with the relevant experiments and results for measuring median relative error, are provided in `code/approximations`. Separately, a Jupyter notebook containing the results included for Operon within the paper is provided in `code/operon`. Lastly, files needed for locally running the Operon experiments and analysis notebook can be installed using the following instructions.

## Installation instructions

The following is only necessary when executing the relevant scripts locally. **Note that the results included within the paper can be viewed without installation by simply opening the relevant CSV or Jupyter notebook files.**

Also, when executing locally, note that only Linux operating systems are likely supported. The following was verified with **insert CPU information** and **insert OS information**.

### Prerequisites
- Ensure that some Conda package management system (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) is installed on the relevant machine.
- Download the latest software release from GitHub, available [here](https://github.com/christophercrary/conference-ppsn-2024/releases/tag/v0.1.1). Extract and paste the contents of the `data.zip` file from the software release into the `code/operon` folder.

Upon doing the above, set up the relevant Conda environment and install the relevant tools by executing the following within a shell program, after having navigated to the repository directory within the shell:

```
conda env create -f environment.yml
conda activate conference-ppsn-2024
bash install.sh
```

### Running the Operon experiments

In order to run the relevant Operon experiments, first navigate to the `code/operon` folder. Then, execute the following commands within your shell program:

```bash
 for backend in {Eigen,Vdt,Stl,Mad_Transcendental_Fast,Mad_Transcendental_Faster,Mad_Transcendental_Fastest}; do ./operon_experiment.py --bin ./build_${backend}/cli/operon_nsgp --data ./problems/ --reps 100 >> my_results.csv; done
```

After the above finishes execution, **insert details about running other relevant scripts**.

**Note:** some of the results reported in the paper will definitely be irreproducible, e.g., runtime and energy measurements. In addition, if there are discrepancies in hardware/firmware, it is possible that there may exist other differences in results, e.g., due to differing implementations of the IEEE-754 floating-point standard.

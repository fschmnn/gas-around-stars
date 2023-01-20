# Stellar associations powering Hɪɪ regions

*last updated 2023.01.20*

this repository contains the code for the two papers

* Stellar associations powering Hɪɪ regions – I. Defining an evolutionary sequence (submitted). 

  ![cutouts](https://raw.githubusercontent.com/fschmnn/cluster/master/references/cutouts.png)

* Stellar associations powering Hɪɪ regions – II. Escape fractions (in preparation).



## Content

* **The package**: often re-used functions are moved to the `src` folder and constitute the main package.
* **Scripts**: the main analyze steps are bundles together in scripts. Here are a few examples: 
  * *match_catalogues.py*: the main script that matches the spatial masks of the HII regions to the spatial masks of the stellar associations.
  * *measure_EW.py*: measure the equivalent width of the HII regions from their spectra (for H$\alpha$ and H$\beta$).
  * *neighboring_associations.py*: find all associations that are in a given radius around each HII region.
* **Notebooks**: jupyter notebooks are then used to post-process the results and create figures.
  * *Project2 Clusters+HII-Regions Single.ipynb*: analyze a single galaxy and produce some intermediate catalogues. 
  * *Project2 Clusters+HII-Regions Multi.ipynb*: read in the previously constructed catalogues and analyze all galaxies at once.
  * *Project3 Escape fractions.ipynb*: 



## Installation

In principle one could clone this repository from [github](https://github.com/fschmnn/cluster) and use it right away. However to ensure that everything works as intended, a few additional steps are recommended.

1. **Set up conda environment**: It is highly advised to run data science projects in a dedicated environment. This has the advantage that any third party packages have the correct version installed which helps to make the results reproducible. We use *conda* to do this. The required packages are listed in `environment.yml` and a new environment, called `pymuse` is created with

   ```bash
   conda env create -f .\environment.yml
   ```

    Every time one opens a new shell, the environment must be activated with

   ```bash
   conda activate pymuse
   ```

   New packages can either be installed by altering the installation file and running

   ```bash
   conda env update -f environment.yml --prune
   ```

   or by typing

   ```bash
   conda install photutils -c astropy
   ```

   Both cases require an active environment. Lastly, a useful addition when working with *jupyter notebooks* are extensions which can be activated with

   ```bash
   conda install -c conda-forge jupyter_contrib_nbextensions
   conda install -c conda-forge jupyter_nbextensions_configurator
   ```

   The extensions can then be activated in the `Nbextensions` tab of the jupyter explorer

2. **Install astrotools**: this package relies on a few functions that were outsourced to a separate package ([astrotools](https://github.com/fschmnn/astrotools)) that should be installed first.

3. **Install the package**: with the dependencies installed, we still need to setup the actual package. To develop the package, simply type

   ```bash
   python setup.py develop
   ```

And that's it. You may have noticed that the project already contains folders and files for unit test and documentations. However neither are currently used but both should eventually be added.


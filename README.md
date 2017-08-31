# Model notebooks for the fragment decomposition regression

Source code and data files for the paper "Measuring and Predicting Sooting Tendencies of Oxygenates, Alkanes, Alkenes, Cycloalkanes, and Aromatics on a Unified Scale"

Model code and data for the fragment decomposition.
`fragdecomp` is a python module I use in a few of the files with different utility functions. All the RDKit magic happens in `fragdecomp.fragment_decomposition`, but its pretty basic.

The notebooks are in this root directory, results of the regression are stored in `data/`

I try to save the interesting results as `csv` files, while I save some of the more intermediate results as pandas pickled dataframes. (I.e., raw distributions of the different fits).

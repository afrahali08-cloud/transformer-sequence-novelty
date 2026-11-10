# Public dataset

Alcock, R. (1999). **Synthetic Control Chart Time Series** [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C59G75](https://doi.org/10.24432/C59G75).

License is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included `synthetic_control.data` retains the original downloaded bytes. It contains 600 simulated control charts, each with 60 real-valued measurements.

The source defines labels by row order, with 100 examples per class. In zero-based row indices these are normal (0–99), cyclic (100–199), increasing trend (200–299), decreasing trend (300–399), upward shift (400–499), and downward shift (500–599).

The project uses a seeded, class-stratified 60/20/20 split of the first five classes and withholds all 100 downward-shift examples for novelty evaluation. This is a custom split of the UCI source and differs from the UCR archive's benchmark split.

Download the source archive from [UCI](https://archive.ics.uci.edu/static/public/139/synthetic+control+chart+time+series.zip). Each run records the SHA-256 of the included data and the exact zero-based split indices in its metrics file.

Source documentation describes the generator in Alcock and Manolopoulos (1999), *Time-Series Similarity Queries Employing a Feature-Based Approach*. These are publicly sourced simulated signals, not experimentally measured sensor or diffraction traces.

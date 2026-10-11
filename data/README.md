# Demonstration feature records

`example_features.csv` contains four development records from the HSP-S source,
restricted to the exact 436 scalar audio columns accepted by the model. There are
no artist names, chart ranks or labels in the prediction inputs.

Upload this file in the app to verify the interface. These examples were used in
training; their predictions are demonstrations, not held-out accuracy evidence.
They are whole-recording Essentia features, not 15-second librosa chorus features.

Source: Michael Vötter, Maximilian Mayerl, Günther Specht and Eva Zangerle (2021),
*Novel Datasets for Evaluating Song Popularity Prediction Tasks*,
https://zenodo.org/records/5383858, CC BY 4.0. Changes: four-row subset, scalar
feature selection and CSV export without metadata or labels. The complete source
can be obtained through the checksum-verified download command in the README.

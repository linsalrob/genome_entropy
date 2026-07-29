Machine-learning workflow
=========================

Purpose and interpretation
--------------------------

The optional ML workflow predicts whether each called ORF matches a CDS in the
input GenBank annotation (``in_genbank``). It is not a functional annotator, and
its probability is not biological proof. Labels inherit the limitations of the
GenBank matching heuristic and source annotation.

Install the optional dependencies with ``pip install "genome_entropy[ml]"``.
This adds XGBoost, scikit-learn, and NumPy. PyTorch is already a core dependency.

Models
------

``xgboost`` is the default. ``XGBoostModel`` calls ``xgboost.train`` with
``objective=binary:logistic``, histogram trees, a default 100 boosting rounds,
maximum depth 6, and learning rate 0.1. It is gradient-boosted XGBoost; the code
does **not** use ``XGBRFClassifier``. Gain importance is normalised to sum to one
and describes the trained model, not causality or biological importance.

``neural_net`` is a two-hidden-layer PyTorch feed-forward classifier with ReLU,
dropout, sigmoid output, binary cross-entropy, and Adam optimisation. It provides
no feature importance.

Feature matrix and missing values
---------------------------------

Features occur in this exact order:

#. ``dna_entropy``
#. ``protein_entropy``
#. ``three_di_entropy``
#. ``dna_length``
#. ``protein_length``
#. ``three_di_length``
#. ``start``
#. ``end``
#. ``strand_plus``
#. ``frame``
#. ``has_start_codon``
#. ``has_stop_codon``
#. ``twelve_state_entropy``
#. ``twelve_state_length``

Unified and legacy pipeline JSON are supported. Missing 12-state values are
represented as NumPy ``NaN``. XGBoost handles these missing values natively.
The neural network learns column medians from its training set, substitutes zero
for an all-missing column, stores those imputation values, and reuses them for
evaluation and prediction.

Splitting and reproducibility
-----------------------------

Exactly one training input mode is required:

* ``--json``: one multi-record JSON file; top-level records are split so all
  ORFs from a genome/sequence remain together. ``--test-split`` defaults to 0.1.
* ``--json-dir``: all ``.json`` and ``.json.gz`` files are loaded, then ORF
  samples are randomly split. This can leak genome-specific patterns across
  train and test and should not be used to claim generalisation to new genomes.
* ``--split-dir``: files are reproducibly shuffled and split 80/20 by file.
  ``--test-split`` is ignored. An optional ``--json-output`` report records file
  lists, parameters, metrics, per-ORF predictions, and feature importance.

The train/test shuffle uses ``--random-seed`` (default 42). The XGBoost model's
internal train/validation split currently uses NumPy's global random permutation
and is not seeded by this option; the neural network also does not seed all
PyTorch operations. Consequently the outer split is reproducible, but complete
training determinism is not guaranteed.

``--validation-split`` defaults to 0.2 and is taken from the training portion.
Avoid sample-level splitting when evaluating genome-level generalisation.

Training, evaluation, and persistence
-------------------------------------

.. code-block:: bash

   genome_entropy ml train --json genomes.json --output model.ubj
   genome_entropy ml train --split-dir results/ --output model.ubj \
       --json-output evaluation.json --random-seed 42
   genome_entropy ml train --json-dir results/ --output model.pt \
       --model-type neural_net --device cuda

Evaluation reports accuracy, precision, recall, F1, AUC, and confusion-matrix
counts. Scikit-learn supplies AUC. XGBoost saves its native model at the requested
path plus a Python-pickle ``.meta`` sidecar containing hyperparameters;
``GenbankClassifier`` also writes a ``.classifier`` pickle with model type and
feature names. Neural-network persistence uses a PyTorch checkpoint plus the
classifier metadata. Treat pickle files as trusted input only.

Prediction
----------

.. code-block:: bash

   genome_entropy ml predict --json results.json --model model.ubj \
       --output predictions.tsv

Exactly one of ``--json`` and ``--json-dir`` is required. The TSV columns are:

``input_id``, ``orf_id``, ``predicted_label``, ``prob_not_in_genbank``,
``prob_in_genbank``, and ``in_genbank``.

``prob_in_genbank`` is the model's positive-class probability and
``predicted_label`` uses a strict threshold greater than 0.5. ``in_genbank`` is
the available source label, not a second prediction.

Devices
-------

The neural network follows PyTorch device support, including CUDA, Apple MPS,
and ROCm (reported through the ``cuda`` API). XGBoost GPU training requires an
XGBoost build compiled with CUDA support and ``--device cuda``. PyTorch ROCm
availability does not imply that the installed XGBoost build can use an AMD GPU;
use CPU XGBoost unless that installation is independently verified.

For XGBoost, automatic detection currently consults ``torch.cuda.is_available``.
An accelerator visible to PyTorch may therefore be selected even when XGBoost
lacks a compatible GPU backend; pass ``--device cpu`` in that case.

Reference
---------

For XGBoost, cite Chen and Guestrin, "XGBoost: A Scalable Tree Boosting
System", KDD (2016), doi:10.1145/2939672.2939785.

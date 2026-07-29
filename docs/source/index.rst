genome_entropy
==============

``genome_entropy`` is a bioinformatics pipeline for measuring information
content across representations derived from genomic DNA:

.. code-block:: text

   DNA → ORFs → proteins → 3Di + optional 12-state structural encodings → entropy

Current multitask ModernProst models emit both 3Di and 12-state (``12st``)
encodings. Legacy ModernProst and ProstT5 models emit 3Di only. Raw Shannon
entropy is calculated for available representations; normalised entropy is a
downstream derived value and is not stored in standard JSON.

The pipeline calls the external ``get_orfs`` program for six-frame ORF discovery,
uses ``pygenetic-code`` for translation, and supports FASTA or GenBank input.
GenBank matching marks whether a called ORF corresponds to an annotated CDS; it
does not assign biological function.

Start here
----------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   user_guide
   models
   data_formats
   ml
   hpc

Reference
---------

.. toctree::
   :maxdepth: 2

   cli
   api
   token_estimation

Development
-----------

.. toctree::
   :maxdepth: 2

   development
   changelog

Project status and support
--------------------------

The package is alpha software. The source repository, issue tracker, and
licence are available at https://github.com/linsalrob/genome_entropy. Include
the package version and accelerator details in bug reports, and do not publish
credentials or sensitive sequence data.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

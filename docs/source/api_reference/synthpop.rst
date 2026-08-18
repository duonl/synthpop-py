API reference
================

This section provides the API reference for the Python modules that make up the
data synthesis workflow. For more in-depth information, and synthpop-py in practe, please see the 
`User Guides <../user_guides/user_guides_index.html>`__
and `Examples <../examples/examples_index.html>`__.

The reference is organised by functionality to make it easier to find the
components you need. The `Synthesiser class <synthesiser_class/synthesiser.html>`_
provides the main interface for generating synthetic data, while the
`Data Preparation <data_processing/Encoder.html>`_ components support the encoding
and handling of missing values required during preparation. The available
`Methods <synthesis_methods/Base_Synth.html>`_ implement different approaches to
data synthesis, and the `Utility Metrics <utility_metrics/S_pMSE.html>`_ can be
used to assess the similarity and utility of synthetic data.

Additional documentation is provided for
`Plotting <plotting/Univariate.html>`_ to help visualise results and internal
`Reproducibility <reproducibility/reproducibility.html>`_ to support consistent
and repeatable synthesis workflows.

Use the sections below to explore the available classes, methods, and
utilities, including their parameters, return values, and usage details.

Table of contents
==================

.. toctree::
   :maxdepth: 4
   :caption: Synthesiser class

   synthesiser_class/synthesiser

.. toctree::
   :maxdepth: 4
   :caption: Data Preparation

   data_processing/Encoder
   data_processing/Missing_value

.. toctree::
   :maxdepth: 4
   :caption: Methods

   synthesis_methods/Base_Synth
   synthesis_methods/CART
   synthesis_methods/Copy
   synthesis_methods/Sample

.. toctree::
   :maxdepth: 4
   :caption: Utility Metrics

   utility_metrics/S_pMSE

.. toctree::
   :maxdepth: 4
   :caption: Plotting

   plotting/Univariate
   plotting/S_pMSE

.. toctree::
   :maxdepth: 4
   :caption: Reproducibility

   reproducibility/reproducibility

CART synthesis
---------------------------------------
.. warning::

   The module for **TreeClassifierMethod** and **TreeRegressorMethod** is not part of the stable public API.
   Interfaces may change without notice.

.. note:: 

   A warning is emitted when more than 25% of the observations in a feature used to train CART belong to rare categories.
   Features with rare categories can increase the risk of :ref:`unintended attribute disclosure <612-attribute-disclosure>`. See :doc:`the examples <../../examples/rare_categories>` for an illustration of this privacy risk.
   The threshold for when a category is considered rare (and emitting this warning) can be adjusted using :func:`tune_cart`. See :ref:`the example on the tune_cart function <../../examples/tune_cart_function>` for details.


.. automodule:: synthpop.methods.cart_synth
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

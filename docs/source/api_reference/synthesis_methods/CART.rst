CART synthesis
---------------------------------------
.. warning::

   The module for **TreeClassifierMethod** and **TreeRegressorMethod** is not part of the stable public API.
   Interfaces may change without notice.

.. note::

   A warning is emitted when CART is trained on a feature that consist for more than half of rare categories.
   If the features used to fit a CART model contain categories with rare values, there is a risk of :ref:`unintended attribute disclosure <612-attribute-disclosure>`.
   See :doc:`the examples <../../examples/rare_categories>` for an example of this privacy problem.
   Use :meth:`tune_cart` to adjust this warning.
   See <TODO: link to tune_cart examples> for how to adjust when this warning is emitted.


.. automodule:: synthpop.methods.cart_synth
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

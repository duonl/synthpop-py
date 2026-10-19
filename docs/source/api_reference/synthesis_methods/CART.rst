CART synthesis
---------------------------------------
.. currentmodule:: synthpop.methods.cart_synth

.. note:: 

   A warning is emitted when more than 25% of the observations in a feature used to train CART belong to rare categories.
   Features with rare categories can increase the risk of :ref:`unintended attribute disclosure <612-attribute-disclosure>`. See :doc:`the examples <../../examples/rare_categories>` for an illustration of this privacy risk.
   The threshold for when a category is considered rare (and emitting this warning) can be adjusted using :func:`tune_cart`. See :doc:`the example on the tune_cart function <../../examples/tune_cart_function>` for details.

.. autoclass:: CartMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. autofunction:: tune_cart

.. note::
   Direct use of these classes is only necessary for advanced use cases that require
   custom configuration of CART's underlying components.
   For the recommended way to configure CART, see
   `User Guide 3.1.4 <../../user_guides/3_synthesis_methods.html#configuring-cart>`__
   and `Example: Configure CART <../.../examples/configure_cart_directly.html>`__.
   




.. _tree-classifier-method:

.. autoclass:: TreeClassifierMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. _tree-regressor-method:

.. autoclass:: TreeRegressorMethod

CART synthesis
---------------------------------------
.. currentmodule:: synthpop.methods.cart_synth

.. autoclass:: CartMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. warning::
   **TreeClassifierMethod** and **TreeRegressorMethod** are not part of the stable public API, 
   as they are a submodule of CART.
   Interfaces may change without notice. 
   See `User Guide 3.1.4. <../../user_guides/3_synthesis_methods.html#configuring-cart>`__
   and `Example: Configure CART <../.../examples/configure_cart_directly.html>`__
   for the recommended way to configure CART's TreeClassifierMethod and TreeRegressorMethod.
   Using TreeClassifierMethod and TreeRegressorMethod directly is only needed for limited specific usecases.

.. _tree-classifier-method:

.. autoclass:: TreeClassifierMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. _tree-regressor-method:

.. autoclass:: TreeRegressorMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. warning::

   The module for **LeafNodeSampling** is not part of the stable public API.
   Interfaces may change without notice.
   Most users do not need to use this class directly, as it is a submodule of CART.

.. automodule:: synthpop.methods.tree_utils
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin
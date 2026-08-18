CART synthesis
---------------------------------------
.. currentmodule:: synthpop.methods.cart_synth

.. autoclass:: CartMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. warning::
   **TreeClassifierMethod** and **TreeRegressorMethod** are not part of the stable public API.
   Interfaces may change without notice. See `User Guide 3.1.4. <../../user_guides/3_synthesis_methods.html#configuring-cart>`__
   for the recommended way to configure CART's TreeClassifierMethod and TreeRegressorMethod.
   Using TreeClassifierMethod and TreeRegressorMethod directly is only needed for limited specific usecases.

.. autoclass:: TreeClassifierMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. autoclass:: TreeRegressorMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. warning::

   The module for **LeafNodeSampling** is not part of the stable public API.
   Interfaces may change without notice.
   Most users do not need to use this class directly.

.. automodule:: synthpop.methods.tree_utils
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin
CART synthesis
---------------------------------------
.. currentmodule:: synthpop.methods.cart_synth

.. autoclass:: CartMethod
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin

.. warning::
   **TreeClassifierMethod** and **TreeRegressorMethod** are not part of the stable public API.
   They are internal components of CART, and their interfaces may change without notice.
   For the recommended way to configure CART, see
   `User Guide 3.1.4 <../../user_guides/3_synthesis_methods.html#configuring-cart>`__
   and `Example: Configure CART <../.../examples/configure_cart_directly.html>`__.
   Direct use of these classes is only necessary for advanced use cases that require
   custom configuration of CART's underlying components.

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

   The **LeafNodeSampler** module is not part of the stable public API and should
   generally not be used directly. It is an internal component of CART,
   and its interface may change without notice.

.. automodule:: synthpop.methods.tree_utils
   :members:
   :show-inheritance:
   :inherited-members: BaseEstimator,_MetadataRequester,_SetOutputMixin
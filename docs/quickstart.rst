Quick Start
===========

Install (core is numpy + scipy only):

.. code-block:: bash

   pip install otwin

Strong end — an analytic port-Hamiltonian system from the catalog:

.. code-block:: python

   import numpy as np
   from otwin import DigitalTwin, evaluate
   from otwin.systems import water_tank

   twin = DigitalTwin(model=water_tank())

   x0 = np.array([1.0])
   t = np.linspace(0, 10, 100)
   u = np.zeros((100, 1))
   forecast = twin.forecast(x0, t, u)
   print(forecast["x"].shape)            # (100, 1)

Evaluate with a leakage-free protocol and mandatory baselines:

.. code-block:: python

   # `data` is your measured series, shape (n_samples, n_features)
   rng = np.random.default_rng(0)
   data = forecast["x"] + rng.normal(0, 0.01, forecast["x"].shape)

   report = evaluate(twin, data, protocol="rolling_origin")
   print(report)                         # skill score vs naive baseline, first

Light end — empirical-law prior + bounded residual + conformal bands — is shown
end to end in ``examples/battery_soh``; a grid-scale dispatch application
(predictive maintenance + real-time optimization) is in
``examples/grid_storage_dispatch``.

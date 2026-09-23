.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======================
prediction-markets-bot
======================


    A multi-venue prediction-market tracker: compare market-implied outcomes against
    independent forecasts, and measure — honestly — whether that produces an edge.


It currently covers temperature markets on **Polymarket** and **Kalshi**, but the name is
deliberately venue- and subject-neutral: nothing in the measurement discipline below is specific
to weather.

The system collects market snapshots and forecasts on a schedule, grades resolved markets against
the *same source the venue actually settles on*, and scores its own forecasts against the market
price. Everything runs in GitHub Actions; git is the datastore.


What it does
============

- **Collects** market snapshots, venue price history, and numerical-weather-prediction forecasts
  (deterministic, multi-model and 122-member ensemble) into append-only, day-partitioned CSVs.
- **Calibrates** forecasts per city and per lead time, then turns them into a probability for
  every market bin.
- **Grades** resolved markets against settlement-faithful truth — the named station reading the
  oracle actually reads, not a nearby grid cell.
- **Scores** itself continuously: Brier and CRPS against the market price and against a raw
  ensemble baseline, with pre-registered sample-size gates.


The honest-measurement discipline
=================================

This is the part worth copying, and most of the repository's history is about it.

**Nothing trades real money until its own pre-registered gate passes.** Gates are committed in
advance precisely so they cannot be moved afterwards to fit a result.

**Every measurement change so far has made the model look worse and the market look better.**
Grading against the wrong ruler — a forecast grid instead of the station, the NWS CLI instead of
the page the market resolves on, routine METARs instead of the full observation set — inflated
results every single time. Assume the next correction costs you too.

**The dominant bug class is silent**: a green run, a plausible number, and no error. A guard that
reads nothing reports the empty set as health; an aggregate agreement rate cannot detect a
one-sided error. Read the disagreement list, not the rate.

**Current verdict: the forecasting model does not beat the market** (Brier 0.166 vs 0.128), so
the model book is off. Live work is on model-free structural strategies and cross-venue basis,
each in forward paper trials behind its own gate.


Layout
======

The active code lives in ``src/polymarket_weather/`` — a legacy name from when the project
tracked one venue. ``src/raincheck/`` is an empty PyScaffold skeleton, kept because the Sphinx
config still points at it.

``CLAUDE.md`` is the real architecture document: data flow, module responsibilities, the
resolution-anchor model, and a long list of incidents with what each one taught. Read it before
changing anything that touches grading or truth.


Quick start
===========

.. code-block:: bash

    pip install -r requirements.txt
    cd src/polymarket_weather

    python main.py                  # fetch, summarise, plot
    python evaluate_oos.py          # the arbiter: model vs market vs ensemble
    python audit_settlements.py     # grading vs actual settlements (must stay >= 95%)

A fresh clone has the committed perishable data but none of the refetchable archives, so
evaluation will report zero gradable markets until you run ``fetch_historical_truth.py`` and
``fetch_station_obs.py``. That is expected, not data loss.

Tests run from the repository root:

.. code-block:: bash

    pytest -o addopts="" tests/ -v


Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.

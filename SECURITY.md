# Security

## What this repository is

A tutorial: a dataset, three notebooks, and a script that produces numbers. It
is not a service, it holds no credentials, and it processes no user input.

The realistic risk surface is therefore small, and worth stating plainly rather
than dressing up:

- **It runs arbitrary computation on your machine** when you execute a notebook,
  like any notebook.
- **The Colab badges fetch data over HTTPS** from `raw.githubusercontent.com` in
  this repository. Nothing else is downloaded.
- **Its dependencies** are NumPy, SciPy, pandas, scikit-learn, Matplotlib and
  Seaborn. Anything you would report about them belongs upstream.

## Reporting

If you find something here that a person could be harmed by — a dependency
pinned to a known-vulnerable version, a notebook cell that would execute
untrusted content, anything of that shape — email **javier@jmarin.info** rather
than opening a public issue, and expect a reply within a week.

For anything that is a *correctness* problem rather than a security one — a
wrong number, a claim the code does not support, a notebook that fails — please
open a public issue instead. Those are more valuable in the open, and this
project would rather be corrected loudly.

## What this file deliberately does not claim

An earlier version of this file described manifest schema validation. This
repository has no such thing; that text belonged to
[`otwin-spec`](https://github.com/otwin-core/otwin-spec) and was copied here by
mistake. A security policy describing a control that does not exist is worse
than no policy, so it is recorded here as a correction rather than silently
deleted.

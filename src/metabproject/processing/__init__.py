"""Processing pipeline steps.

Each function takes and returns a DataFrame; I/O lives elsewhere. The fixed
pipeline order (see ``specs/process.md``) is:

    prevalence filter → imputation → weight correction → normalization →
    transformation → scaling
"""

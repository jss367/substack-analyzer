"""Frequently asked questions content for the Help tab."""

FAQ_ITEMS = [
    {
        "question": "What is this value?",
        "answer": (
            "Key metrics shown in the app and what they mean:\n"
            "- **Ending free / Ending premium**: Subscribers in each cohort in the last simulated month.\n"
            "- **Net MRR**: Monthly recurring revenue after Substack and Stripe fees from monthly subscribers.\n"
            "- **Cumulative profit**: Sum of monthly profit (net revenue minus ad spend and ad manager fee) over time.\n"
            "- **Cumulative ad spend**: Total advertising spend across all simulated months.\n"
            "- **ROAS (net revenue / ad spend)**: Net revenue divided by total ad spend, showing revenue efficiency of ads.\n"
            "- **Blended CAC (paid only)**: Total ad spend divided by the number of new free subscribers attributed to ads.\n"
            "- **Payback month (cumulative)**: First month where cumulative profit turns positive.\n"
            "- **Ad manager fee**: Monthly management fee applied only in months with ad spend; it reduces monthly profit."
        ),
    },
    {
        "question": "How does the simulator move free subscribers to paid (and vice versa)?",
        "answer": (
            "The simulator only grows the free audience directly—through an organic month-over-month rate "
            "and any paid acquisition you fund (ad spend divided by CAC, modified by the ad-response curve "
            "and carrying-capacity damping). Paid subscribers increase by converting a share of that free "
            "audience: a percentage of the current month’s new free signups convert immediately, and a "
            "smaller ongoing percentage of the existing free base converts every month. Those conversions "
            "are added to the premium pool while being removed from the free pool. Premium churn is then "
            "applied monthly to the premium pool, so net premium growth equals conversions minus churn—"
            "there’s no standalone premium growth rate, just conversion from free plus churn effects."
        ),
    },
    {
        "question": "When does the Help tab update its growth equation?",
        "answer": (
            "When you run Stage 4 of the Estimators tab, the fitted piecewise logistic equation is stored in "
            "session state and displayed here. If you have not fit a model yet, the Help tab shows the default "
            "deterministic cohort equations used by the Simulator sidebar."
        ),
    },
    {
        "question": "What should I do before loading a session bundle?",
        "answer": (
            "If you import new data, first download the current bundle from Save / Load to avoid losing work. "
            "When you upload a bundle, the app restores events, fits, and simulator inputs, and will switch to "
            "the Simulator tab automatically."
        ),
    },
]

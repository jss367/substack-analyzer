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
            "Free cohorts upgrade to paid through two conversion rates in the deterministic equations: "
            "`p_new` (new-subscriber premium conversion) applies to brand-new free signups each month, "
            "and `p_ongoing` (ongoing premium conversion) applies to the existing free base. The conversions "
            "subtract from free and add to paid in the same month. Paid does not downgrade back to free; "
            "instead, churn rates (`c_f`, `c_p`) remove subscribers from each cohort."
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

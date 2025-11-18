"""Frequently asked questions content for the Help tab."""

FAQ_ITEMS = [
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

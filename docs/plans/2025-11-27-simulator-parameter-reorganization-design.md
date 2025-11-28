# Simulator Parameter Reorganization - Design Document

**Date:** 2025-11-27
**Status:** Approved
**Goal:** Reorganize simulator sidebar parameters with mirrored Free/Premium structure

---

## Problem Statement

The current simulator sidebar has inconsistent organization:
- Carrying capacity parameters appear on the main page instead of the sidebar
- Free/premium parameters are scattered across multiple expanders ("Starting point", "Growth & churn")
- With dual-series calibration inferring separate parameters, the scattered organization is confusing

---

## Design Goals

1. Create clean, intuitive sidebar structure with mirrored "Free parameters" and "Premium parameters" expanders
2. Make it obvious which parameters affect which subscriber type
3. Move all parameters from main page to sidebar for consistency
4. Support dual-series calibration workflow where inferred values populate both free and premium parameters
5. Use progressive disclosure (essential parameters expanded, advanced collapsed)

---

## New Sidebar Structure

### Complete Organization (top to bottom)

**1. Horizon** *(expanded by default)*
- Months to simulate (slider: 12-24 months)

**2. Dual-series fit info message** *(conditional)*
- Displays when dual_fit exists in session state
- Message: "Using inferred parameters from Phase 1 dual-series fit as defaults. You can override them below."

**3. Free parameters** *(expanded by default)*
- Starting free subscribers
- Organic monthly growth (free)
- Monthly churn (free)
- Carrying capacity (free) - optional

**4. Premium parameters** *(expanded by default)*
- Starting premium subscribers
- Monthly churn (premium)
- Carrying capacity (premium) - optional

**5. Conversions** *(collapsed by default - existing expander)*
- New-subscriber premium conversion
- Ongoing premium conversion of existing free

**6. Downgrades** *(collapsed by default - NEW expander)*
- Paid downgrades to free

**7. Ad spend controls** *(collapsed by default - existing)*
- Free ad spend schedule
- Premium ad spend schedule
- Ad manager monthly fee

**8. Acquisition costs** *(collapsed by default - existing)*
- Cost per new free subscriber
- Cost per new premium subscriber
- Diminishing returns parameters (adstock_lambda, ad_log_theta)

**9. Pricing & fees** *(collapsed by default - existing)*
- Premium monthly price
- Annual pricing options
- Substack and Stripe fees

---

## Parameter Migration Map

### From "Starting point" expander (REMOVE THIS EXPANDER)
- Starting free subscribers → **Free parameters** expander
- Starting premium subscribers → **Premium parameters** expander

### From "Growth & churn" expander (REMOVE THIS EXPANDER)
- Organic monthly growth (free) → **Free parameters** expander
- Monthly churn (free) → **Free parameters** expander
- Monthly churn (premium) → **Premium parameters** expander
- Paid downgrades to free → **Downgrades** expander (NEW)

### From main page (currently orphaned around line 1910-1923)
- Free carrying capacity (optional) → **Free parameters** expander
- Premium carrying capacity (optional) → **Premium parameters** expander

### Stays in place (just reordered)
- Horizon expander
- Conversions expander
- Ad spend controls
- Acquisition costs
- Pricing & fees

---

## Benefits

### 1. Clear Mental Model
Users think in terms of "free subscriber behavior" and "premium subscriber behavior" as separate but related systems. Mirrored structure makes it obvious both types have similar dynamics.

### 2. Dual-Series Integration
When dual-series fit infers parameters, the info message appears right before Free/Premium expanders. Users see "here are your inferred values" followed by the exact parameters that were inferred, all grouped together.

### 3. Progressive Disclosure
- Essential parameters (horizon, free dynamics, premium dynamics) visible by default
- Advanced relationships (conversions, downgrades) collapsed until needed
- Cost parameters collapsed until needed
- Reduces cognitive load for new users

### 4. Consistent Parameter Location
Every parameter has an obvious home in the sidebar. No orphaned inputs on main page. If looking for free parameter → check Free; if premium → check Premium.

### 5. Scalability
If we add more free/premium-specific parameters (e.g., different growth rates), the structure naturally accommodates them.

---

## Implementation Details

### Code Location
Primary changes in `app.py`, `sidebar_inputs()` function (starts around line 1447)

### Changes Required

**1. Remove old expanders:**
```python
# DELETE these blocks:
with st.sidebar.expander("Starting point", expanded=True):
    # starting free/premium

with st.sidebar.expander("Growth & churn", expanded=True):
    # organic growth, churn rates, downgrades
```

**2. Create new Free parameters expander:**
```python
with st.sidebar.expander("Free parameters", expanded=True):
    start_free = number_input_state(
        "Starting free subscribers",
        min_value=0,
        default_value=int(_get_state("start_free", 0)),
        step=10,
        key="start_free",
    )
    organic_growth = number_input_state(
        "Organic monthly growth (free)",
        min_value=0.0,
        max_value=1.0,
        default_value=float(_get_state("organic_growth", DEFAULT_GROWTH_RATE)),
        step=0.001,
        format="%0.3f",
        key="organic_growth",
    )
    churn_free = number_input_state(
        "Monthly churn (free)",
        min_value=0.0,
        max_value=1.0,
        default_value=float(inferred_churn_free) if inferred_churn_free is not None else float(_get_state("churn_free", 0.01)),
        step=0.001,
        format="%0.3f",
        key="churn_free",
    )
    capacity_free = number_input_state(
        "Carrying capacity (free) - optional",
        min_value=0.0,
        default_value=float(_get_state("carrying_capacity_free", 0.0)),
        step=100.0,
        key="carrying_capacity_free",
    )
```

**3. Create new Premium parameters expander:**
```python
with st.sidebar.expander("Premium parameters", expanded=True):
    start_premium = number_input_state(
        "Starting premium subscribers",
        min_value=0,
        default_value=int(_get_state("start_premium", 0)),
        step=1,
        key="start_premium",
    )
    churn_prem = number_input_state(
        "Monthly churn (premium)",
        min_value=0.0,
        max_value=1.0,
        default_value=float(inferred_churn_premium) if inferred_churn_premium is not None else float(_get_state("churn_prem", 0.01)),
        step=0.001,
        format="%0.3f",
        key="churn_prem",
    )
    capacity_premium = number_input_state(
        "Carrying capacity (premium) - optional",
        min_value=0.0,
        default_value=float(_get_state("carrying_capacity_premium", 0.0)),
        step=100.0,
        key="carrying_capacity_premium",
    )
```

**4. Create new Downgrades expander:**
```python
with st.sidebar.expander("Downgrades", expanded=False):
    downgrade_to_free = number_input_state(
        "Paid downgrades to free",
        min_value=0.0,
        max_value=1.0,
        default_value=float(_get_state("downgrade_to_free", 0.0)),
        step=0.001,
        format="%0.3f",
        key="downgrade_to_free",
    )
```

**5. Critical: Move carrying capacity from main page**
Lines 1910-1923 currently render on main page. These inputs must be deleted from their current location and placed inside the sidebar expanders as shown above.

**6. Process carrying capacity values:**
```python
# After the expanders, before return statement:
carrying_capacity_free = float(capacity_free) if float(capacity_free or 0.0) > 0 else None
carrying_capacity_premium = float(capacity_premium) if float(capacity_premium or 0.0) > 0 else None
```

### Session State Compatibility
All existing session state keys remain unchanged:
- `start_free`, `start_premium`
- `organic_growth`
- `churn_free`, `churn_prem`
- `carrying_capacity_free`, `carrying_capacity_premium`
- `downgrade_to_free`

This ensures backward compatibility with saved sessions.

### Dual-Series Integration
The dual-fit info message (lines 1493-1502) stays between Horizon and Free parameters expanders, so the flow is:
1. Set time scope (Horizon)
2. See "Using inferred parameters..." message (if dual-fit exists)
3. See/edit Free parameters (with inferred defaults)
4. See/edit Premium parameters (with inferred defaults)

---

## Testing Checklist

- [ ] All parameters render in correct expanders
- [ ] Carrying capacities no longer appear on main page
- [ ] Correct expanders expanded by default (Horizon, Free, Premium)
- [ ] Session state persists across page reloads
- [ ] Dual-series inferred parameters populate correctly
- [ ] Manual parameter overrides work
- [ ] Simulation runs with reorganized parameters
- [ ] No visual regressions in sidebar layout

---

## Future Enhancements (Out of Scope)

- Add tooltips/help text explaining carrying capacity
- Consider adding per-segment growth rates in Free parameters if dual-series provides them
- Visual indicators showing which parameters were inferred vs manually set
- Collapsible "Advanced" subsection within Free/Premium for less common parameters

---

## Approval

**Approved by:** User
**Date:** 2025-11-27

Ready for implementation.

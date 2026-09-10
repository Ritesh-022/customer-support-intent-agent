# Phase 4: Brand Selection

Input: `dataset\twcs\twcs.csv`

Sample size target per brand: 2000 conversations

The metrics below are comparison evidence. They do not automatically select a target brand.

| Brand | Conversations | Customer msgs | Responses | Median turns | P75 turns | Repeat rate | Ambiguous rate | Escalation rate | Action proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AmazonHelp | 2000 | 5100 | 4302 | 3.0 | 5.0 | 2.63% | 11.78% | 2.47% | 47.21% |
| AppleSupport | 2000 | 3278 | 2635 | 2.0 | 4.0 | 1.34% | 8.39% | 0.40% | 84.14% |
| Uber_Support | 2000 | 3312 | 2641 | 2.0 | 3.0 | 2.11% | 7.76% | 2.99% | 87.66% |
| SpotifyCares | 2000 | 3262 | 2978 | 2.0 | 4.0 | 1.75% | 10.42% | 0.12% | 54.40% |
| AmericanAir | 2000 | 3724 | 2800 | 2.0 | 4.0 | 0.67% | 6.36% | 1.53% | 44.25% |

## How to Read This

- Exact repeat rate measures normalized duplicate customer text in the sample; it is not semantic near-duplicate detection.
- Response action proxy counts replies containing an action, resource, private-support instruction, or similar support cue; it is NOT a resolution label.
- Ambiguous messages are short or question-only messages and should be reviewed as possible context-dependent cases.
- Escalation signals are keyword indicators for manual review, not confirmed escalation outcomes.
- Multi-brand conversations are flagged separately and should not be mixed into a single-brand target dataset.

## Decision Log

**Selected brand: AppleSupport**

AppleSupport was selected because it combines:
- The highest response action proxy rate (84.14%) among all candidates, indicating the richest set of actionable historical responses for RAG retrieval.
- The lowest exact customer repeat rate (1.34%), meaning less duplicate/templated noise in the training data.
- The lowest escalation signal rate (0.40%), reducing label noise from high-risk edge cases.
- A large conversation pool with sufficient diversity for intent taxonomy coverage.

**Important caveats:**
- The response action proxy rate is NOT a resolution rate. It counts replies containing action cues (DM links, reset instructions, etc.), not confirmed resolutions.
- Multi-brand conversations are excluded from training and evaluation sets.

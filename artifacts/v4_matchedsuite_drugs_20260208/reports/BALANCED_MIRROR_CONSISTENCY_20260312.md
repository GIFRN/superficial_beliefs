# Balanced Mirror Consistency

Mirror consistency is computed on balanced test runs by pairing each original trial with its slot-preserving mirror.
Choice consistency is measured on the chosen underlying profile rather than the raw A/B label.

## Family x Effort Means
| family | effort | choice_expected_profile_agreement | choice_majority_profile_agreement | premise_expected_attr_agreement | premise_majority_attr_agreement | tau_expected_attr_agreement | tau_majority_attr_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 4.5 | low | 0.854 | 0.880 | 0.759 | 0.802 | 0.773 | 0.849 |
| Claude Haiku 4.5 | minimal | 0.697 | 0.742 | 0.612 | 0.652 | 0.725 | 0.779 |
| GPT-5-mini | low | 0.891 | 0.926 | 0.811 | 0.852 | 0.828 | 0.903 |
| GPT-5-mini | minimal | 0.822 | 0.821 | 0.733 | 0.744 | 0.787 | 0.835 |
| GPT-5-nano | low | 0.825 | 0.878 | 0.692 | 0.767 | 0.654 | 0.755 |
| GPT-5-nano | minimal | 0.755 | 0.779 | 0.605 | 0.658 | 0.532 | 0.639 |
| Ministral-3-14B | low | 0.747 | 0.777 | 0.576 | 0.674 | 0.689 | 0.779 |
| Ministral-3-14B | minimal | 0.748 | 0.784 | 0.580 | 0.687 | 0.689 | 0.781 |
| Qwen3.5-14B | low | 0.630 | 0.629 | 0.560 | 0.565 | 0.722 | 0.742 |
| Qwen3.5-14B | minimal | 0.629 | 0.627 | 0.562 | 0.565 | 0.726 | 0.746 |

## Theme Means
| theme | choice_expected_profile_agreement | choice_majority_profile_agreement | premise_expected_attr_agreement | premise_majority_attr_agreement | tau_expected_attr_agreement | tau_majority_attr_agreement |
| --- | --- | --- | --- | --- | --- | --- |
| drugs | 0.735 | 0.765 | 0.586 | 0.654 | 0.681 | 0.754 |
| placebo_label_border | 0.775 | 0.795 | 0.686 | 0.720 | 0.780 | 0.847 |
| placebo_packaging | 0.766 | 0.792 | 0.685 | 0.721 | 0.755 | 0.798 |
| policy | 0.786 | 0.816 | 0.679 | 0.739 | 0.696 | 0.767 |
| software | 0.738 | 0.754 | 0.609 | 0.649 | 0.651 | 0.739 |

## Per-Row Table
| theme | family | effort | choice_expected_profile_agreement | choice_majority_profile_agreement | premise_expected_attr_agreement | premise_majority_attr_agreement |
| --- | --- | --- | --- | --- | --- | --- |
| drugs | Claude Haiku 4.5 | low | 0.844 | 0.876 | 0.691 | 0.773 |
| drugs | Claude Haiku 4.5 | minimal | 0.664 | 0.701 | 0.519 | 0.546 |
| drugs | GPT-5-mini | low | 0.876 | 0.928 | 0.724 | 0.773 |
| drugs | GPT-5-mini | minimal | 0.805 | 0.835 | 0.684 | 0.763 |
| drugs | GPT-5-nano | low | 0.822 | 0.856 | 0.635 | 0.691 |
| drugs | GPT-5-nano | minimal | 0.747 | 0.753 | 0.589 | 0.680 |
| drugs | Ministral-3-14B | low | 0.743 | 0.794 | 0.556 | 0.691 |
| drugs | Ministral-3-14B | minimal | 0.739 | 0.794 | 0.563 | 0.680 |
| drugs | Qwen3.5-14B | low | 0.557 | 0.557 | 0.454 | 0.474 |
| drugs | Qwen3.5-14B | minimal | 0.555 | 0.557 | 0.447 | 0.464 |
| placebo_label_border | Claude Haiku 4.5 | low | 0.862 | 0.907 | 0.759 | 0.763 |
| placebo_label_border | Claude Haiku 4.5 | minimal | 0.696 | 0.701 | 0.645 | 0.619 |
| placebo_label_border | GPT-5-mini | low | 0.885 | 0.897 | 0.844 | 0.876 |
| placebo_label_border | GPT-5-mini | minimal | 0.873 | 0.856 | 0.821 | 0.804 |
| placebo_label_border | GPT-5-nano | low | 0.844 | 0.876 | 0.733 | 0.773 |
| placebo_label_border | GPT-5-nano | minimal | 0.838 | 0.887 | 0.689 | 0.742 |
| placebo_label_border | Ministral-3-14B | low | 0.765 | 0.804 | 0.623 | 0.753 |
| placebo_label_border | Ministral-3-14B | minimal | 0.767 | 0.804 | 0.626 | 0.742 |
| placebo_label_border | Qwen3.5-14B | low | 0.610 | 0.608 | 0.563 | 0.557 |
| placebo_label_border | Qwen3.5-14B | minimal | 0.610 | 0.608 | 0.558 | 0.567 |
| placebo_packaging | Claude Haiku 4.5 | low | 0.868 | 0.907 | 0.790 | 0.835 |
| placebo_packaging | Claude Haiku 4.5 | minimal | 0.729 | 0.794 | 0.648 | 0.722 |
| placebo_packaging | GPT-5-mini | low | 0.917 | 0.948 | 0.870 | 0.887 |
| placebo_packaging | GPT-5-mini | minimal | 0.849 | 0.845 | 0.798 | 0.804 |
| placebo_packaging | GPT-5-nano | low | 0.838 | 0.907 | 0.723 | 0.784 |
| placebo_packaging | GPT-5-nano | minimal | 0.798 | 0.814 | 0.630 | 0.629 |
| placebo_packaging | Ministral-3-14B | low | 0.725 | 0.753 | 0.604 | 0.670 |
| placebo_packaging | Ministral-3-14B | minimal | 0.725 | 0.753 | 0.602 | 0.711 |
| placebo_packaging | Qwen3.5-14B | low | 0.602 | 0.598 | 0.590 | 0.577 |
| placebo_packaging | Qwen3.5-14B | minimal | 0.606 | 0.598 | 0.600 | 0.588 |
| policy | Claude Haiku 4.5 | low | 0.851 | 0.866 | 0.795 | 0.835 |
| policy | Claude Haiku 4.5 | minimal | 0.717 | 0.742 | 0.652 | 0.701 |
| policy | GPT-5-mini | low | 0.872 | 0.918 | 0.786 | 0.835 |
| policy | GPT-5-mini | minimal | 0.831 | 0.856 | 0.721 | 0.753 |
| policy | GPT-5-nano | low | 0.846 | 0.907 | 0.748 | 0.866 |
| policy | GPT-5-nano | minimal | 0.730 | 0.794 | 0.614 | 0.670 |
| policy | Ministral-3-14B | low | 0.793 | 0.825 | 0.552 | 0.649 |
| policy | Ministral-3-14B | minimal | 0.794 | 0.825 | 0.556 | 0.670 |
| policy | Qwen3.5-14B | low | 0.715 | 0.722 | 0.682 | 0.711 |
| policy | Qwen3.5-14B | minimal | 0.709 | 0.711 | 0.685 | 0.701 |
| software | Claude Haiku 4.5 | low | 0.843 | 0.845 | 0.758 | 0.804 |
| software | Claude Haiku 4.5 | minimal | 0.682 | 0.773 | 0.597 | 0.670 |
| software | GPT-5-mini | low | 0.906 | 0.938 | 0.830 | 0.887 |
| software | GPT-5-mini | minimal | 0.750 | 0.711 | 0.642 | 0.598 |
| software | GPT-5-nano | low | 0.776 | 0.845 | 0.623 | 0.722 |
| software | GPT-5-nano | minimal | 0.663 | 0.649 | 0.505 | 0.567 |
| software | Ministral-3-14B | low | 0.709 | 0.711 | 0.545 | 0.608 |
| software | Ministral-3-14B | minimal | 0.717 | 0.742 | 0.553 | 0.629 |
| software | Qwen3.5-14B | low | 0.666 | 0.660 | 0.512 | 0.505 |
| software | Qwen3.5-14B | minimal | 0.664 | 0.660 | 0.520 | 0.505 |
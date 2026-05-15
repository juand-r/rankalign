# membership-sans-rosch-v0 (gemma-2-2b, epoch2) → rosch — per-task gen-ROC × 100

Rows: 10 rosch tasks ordered by item-overlap with the membership training pool (high overlap on top, rosch-sport at the bottom).

Columns: 4 self-eval variants and 4 neg-eval variants of the same set (Base, RankAlign+fsx, SFT+fsx, Offline {self,neg}-TC). Note: every non-Base variant in this cohort includes `--force-same-x`; see [`docs/membership_to_rosch_recipe_inventory.md`](../../docs/membership_to_rosch_recipe_inventory.md).

**Bold = highest value in the row across all 8 columns.** This mixes self and neg eval refs (different metrics on the same checkpoint) — interpret as a visual read, not a rigorous comparison.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

<table>
<thead>
<tr><th rowspan="2">task (overlap)</th><th colspan="4">self</th><th colspan="4">neg</th></tr>
<tr><th>Base</th><th>RankAlign+fsx</th><th>SFT+fsx</th><th>Offline self-TC</th><th>Base</th><th>RankAlign+fsx</th><th>SFT+fsx</th><th>Offline neg-TC</th></tr>
</thead>
<tbody>
<tr><td>rosch-bird (89%)</td><td>70.35</td><td>79.57</td><td><strong>89.50</strong></td><td>87.56</td><td>59.58</td><td>71.15</td><td>84.92</td><td>68.85</td></tr>
<tr><td>rosch-carpenters-tool (61%)</td><td>71.09</td><td>63.98</td><td>73.80</td><td>73.17</td><td>74.36</td><td><strong>77.96</strong></td><td>64.96</td><td>64.43</td></tr>
<tr><td>rosch-fruit (60%)</td><td>76.73</td><td>85.46</td><td>81.15</td><td>91.67</td><td>84.98</td><td><strong>95.46</strong></td><td>82.54</td><td>81.43</td></tr>
<tr><td>rosch-vehicle (56%)</td><td>85.69</td><td>92.30</td><td>86.62</td><td><strong>93.07</strong></td><td>77.04</td><td>77.93</td><td>47.89</td><td>87.98</td></tr>
<tr><td>rosch-furniture (45%)</td><td>80.19</td><td>87.63</td><td>89.34</td><td>93.25</td><td>93.00</td><td><strong>97.66</strong></td><td>94.03</td><td>83.44</td></tr>
<tr><td>rosch-vegetable (44%)</td><td>82.77</td><td>80.31</td><td>80.07</td><td>86.40</td><td><strong>92.10</strong></td><td>89.99</td><td>77.39</td><td>74.74</td></tr>
<tr><td>rosch-toy (42%)</td><td>79.17</td><td>77.16</td><td>79.48</td><td>78.67</td><td>75.54</td><td><strong>81.02</strong></td><td>77.16</td><td>57.14</td></tr>
<tr><td>rosch-clothing (36%)</td><td>69.06</td><td>88.15</td><td>88.71</td><td><strong>90.83</strong></td><td>54.83</td><td>74.66</td><td>66.71</td><td>89.57</td></tr>
<tr><td>rosch-weapon (36%)</td><td>77.72</td><td>75.88</td><td>78.83</td><td>78.54</td><td>86.99</td><td><strong>87.39</strong></td><td>79.37</td><td>73.84</td></tr>
<tr><td>rosch-sport (9%)</td><td>80.41</td><td>85.92</td><td>82.07</td><td><strong>90.57</strong></td><td>78.69</td><td>76.39</td><td>73.68</td><td>81.21</td></tr>
</tbody>
</table>

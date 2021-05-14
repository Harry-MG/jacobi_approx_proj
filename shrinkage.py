def shrinkage(a, kappa):
  y = np.max(0, a-kappa) - np.max(0, -a-kappa);
  return y
"""
Configuration globale pour les modules d'optimisation de portefeuille.
Permet de choisir la méthode d'optimisation (Monte-Carlo ou descente de gradient)
sans modifier les appels dans le serveur.
"""
# Méthode d'optimisation : "monte_carlo", "gradient_fixe" ou "gradient_optimal"
# ("gradient" est accepté comme alias pour "gradient_optimal")
OPTIMIZATION_METHOD = "gradient_optimal"
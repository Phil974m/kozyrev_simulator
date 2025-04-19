import pygame
import numpy as np
import sys

# Paramètres de simulation
RESOLUTION = 256          # Résolution spatiale
SCREEN_SIZE = 512          # Taille de la fenêtre
FPS = 60                   # Images par seconde
DT = 0.05                  # Pas temporel
C = 4.0                    # Vitesse des ondes
TAU_C = 1000.0             # Temps de cohérence
LAMBDA = 0.5               # Couplage non-linéaire
KAPPA = 0.2                # Couplage entropique
DAMPING = 0.999            # Stabilisation numérique

# Initialisation Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()

# Grilles de calcul
x, y = np.meshgrid(np.arange(RESOLUTION), np.arange(RESOLUTION))
psi_real = np.exp(-((x-RESOLUTION//2)**2 + (y-RESOLUTION//2)**2)/(10**2))
psi_imag = np.zeros_like(psi_real)
sigma = np.zeros((RESOLUTION, RESOLUTION))

# Variables historiques pour l'intégration
psi_prev = psi_real + 1j*psi_imag
psi_current = psi_prev.copy()

# Fonction de Laplacien
def laplacian(field):
    return (np.roll(field, +1, 0) + np.roll(field, -1, 0) +
            np.roll(field, +1, 1) + np.roll(field, -1, 1) - 4*field)

def magnitude_to_color(mag):
    hue = (np.angle(psi_current) + np.pi) / (2*np.pi)  # Phase 0-1
    sat = np.ones_like(mag)
    val = mag / np.max(mag) if np.max(mag) > 0 else mag
    hsv = np.stack([hue, sat, val], axis=-1)
    return pygame.surfarray.make_surface((hsv * 255).astype(np.uint8))

# Boucle principale
running = True
drawing = False
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Réinitialiser
                psi_current = np.exp(-((x-RESOLUTION//2)**2 + (y-RESOLUTION//2)**2)/(10**2))
                sigma.fill(0)
    
    # Ajout de sources entropiques
    if drawing:
        mx, my = pygame.mouse.get_pos()
        ix = int(mx * RESOLUTION / SCREEN_SIZE)
        iy = int(my * RESOLUTION / SCREEN_SIZE)
        sigma[iy-3:iy+4, ix-3:ix+4] += 0.5
    
    # Calcul du Laplacien
    lapl = laplacian(psi_current)
    
    # Termes de l'équation
    nonlinear = LAMBDA * np.abs(psi_current)**2 * psi_current
    source = KAPPA * sigma * (1 + 0.5j)  # Source complexe
    damping_term = (1/TAU_C**2) * psi_current
    
    # Intégration temporelle
    psi_next = (2*psi_current - psi_prev + 
               DT**2 * (C**2 * lapl - damping_term + nonlinear + source))
    
    # Mise à jour des variables
    psi_prev, psi_current = psi_current, psi_next * DAMPING
    
    # Visualisation HSV (teinte=phase, saturation=1, valeur=magnitude)
    mag = np.abs(psi_current)
    surf = magnitude_to_color(mag)
    surf = pygame.transform.scale(surf, (SCREEN_SIZE, SCREEN_SIZE))
    
    # Affichage
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    clock.tick(FPS)
    sigma *= 0.85  # Décroissance entropique

pygame.quit()
sys.exit()

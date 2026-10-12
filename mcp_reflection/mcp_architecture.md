# Model Context Protocol (MCP) - Architecture Responsible

## Introduction

Le **Model Context Protocol (MCP)** est un protocole standardisé développé par Anthropic
pour permettre aux LLMs (Large Language Models) d'interagir avec des outils et des données
de manière responsable, sécurisée et contrôlée. Ce document présente notre réflexion sur
l'architecture responsable des agents A2A dans le cadre du projet Smart eCommerce Intelligence.

## Principes Fondamentaux du MCP

### 1. Transparence (Transparency)
Toutes les actions des agents doivent être tracées et explicables.

**Notre implémentation :**
- Chaque agent A2A (ShopifyAgent, WooCommerceAgent) journalise ses actions avec timestamps
- Les décisions ML sont accompagnées de métriques de performance et de feature importance
- Le dashboard Streamlit expose toutes les étapes du pipeline de manière visuelle

### 2. Contrôle et Consentement (Control & Consent)
L'utilisateur doit garder le contrôle sur les actions des agents.

**Notre implémentation :**
- Activation explicite des fonctionnalités LLM via le dashboard
- Filtres interactifs pour la sélection des données
- Validation manuelle possible avant toute action critique
- Paramètres ajustables pour les modèles ML (K, poids, seuils)

### 3. Isolation et Sécurité (Isolation & Security)
Les serveurs MCP ne doivent pas exposer plus que nécessaire.

**Notre implémentation :**
- Les agents de scraping n'exécutent que des requêtes GET read-only
- Aucune modification de données sur les sites scrappés
- Utilisation de données synthétiques par défaut
- Pas de stockage de données sensibles ou PII

### 4. Contextualisation (Context)
Les LLMs doivent recevoir le contexte approprié pour chaque tâche.

**Notre implémentation :**
- Prompts structurés avec contexte clair (dataset, métriques, objectifs)
- Templates de prompts spécialisés par type d'analyse
- Passage des données pertinentes uniquement (pas de données brutes complètes)

### 5. Auditabilité (Auditability)
Toutes les interactions doivent pouvoir être auditées.

**Notre implémentation :**
- Logs complets des appels API et des actions des agents
- Rapports d'analyse exportables (JSON, CSV)
- Traçabilité des décisions ML (feature importance, métriques)
- Historique des sessions dans le dashboard

## Architecture MCP du Projet

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MCP HOST (Application Principale)            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Streamlit Dashboard                         │  │
│  │  - Interface utilisateur                                     │  │
│  │  - Contrôle des autorisations                                │  │
│  │  - Visualisation des résultats                               │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │                    MCP CLIENT (LangChain)                     │  │
│  │                                                               │  │
│  │  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐  │  │
│  │  │Prompt Mgmt  │  │Permission Check│  │  Logging/Audit   │  │  │
│  │  │- Templates  │  │- Rate Limiting │  │  - Full Traces   │  │  │
│  │  │- Versioning │  │- Access Ctrl   │  │  - Error Logging │  │  │
│  │  └─────────────┘  └────────────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │                     MCP SERVERS                               │  │
│  │                                                               │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │  │
│  │  │  Data Server   │  │  Model Server  │  │  Report Server │  │  │
│  │  │                │  │                │  │                │  │  │
│  │  │ - CSV/API      │  │ - Scikit-learn │  │ - Generator    │  │  │
│  │  │ - Synth. Data  │  │ - XGBoost     │  │ - Formatter    │  │  │
│  │  │ - Cache Layer  │  │ - LightGBM    │  │ - Export       │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Composants MCP Détaillés

### 1. MCP Host (Streamlit Dashboard)
- **Rôle** : Interface principale de l'utilisateur avec le système
- **Fonctionnalités** :
  - Navigation et filtrage des données
  - Activation/désactivation des fonctionnalités LLM
  - Configuration des paramètres ML
  - Visualisation interactive des résultats
- **Responsabilités MCP** :
  - Déclarer les intentions à l'utilisateur
  - Obtenir le consentement avant les actions
  - Afficher les logs et traces

### 2. MCP Client (LangChain Agent Wrapper)
- **Rôle** : Orchestrateur des appels LLM avec contrôle
- **Fonctionnalités** :
  - Gestion des prompts avec templates versionnés
  - Vérification des permissions avant chaque appel
  - Rate limiting et quota management
  - Journalisation complète des interactions
- **Responsabilités MCP** :
  - Nettoyer et valider les entrées utilisateur
  - Limiter le contexte partagé avec le LLM
  - Journaliser requêtes/réponses pour audit

### 3. MCP Servers
- **Data Server** : Accès aux données produits (scraping, synthétiques)
- **Model Server** : Hébergement et exécution des modèles ML
- **Report Server** : Génération de rapports et visualisations

### 4. Logs & Permissions
- **Journalisation** : Toute requête est horodatée avec source, action et résultat
- **Permissions** : Accès validé manuellement ou automatiquement selon le risque
- **Sécurité** : Isolation des serveurs, pas de modification externe

## Application des Principes MCP à Nos Agents A2A

### Agent de Scraping Shopify
| Principe MCP | Application |
|-------------|-------------|
| Transparence | Log complet des URLs visitées, données extraites, timing |
| Contrôle | L'utilisateur choisit le mode (synthétique/réel), le nombre max de produits |
| Isolation | Requêtes GET uniquement, respect du robots.txt, rate limiting |
| Auditabilité | Export CSV complet, statistiques de scraping |

### Agent ML Classification
| Principe MCP | Application |
|-------------|-------------|
| Transparence | Feature importance, métriques de performance (précision, rappel, F1) |
| Contrôle | Choix de l'algorithme, paramètres ajustables, train/test split |
| Isolation | Modèles isolés dans des conteneurs séparés |
| Auditabilité | Matrice de confusion, courbes ROC, rapports de classification |

### Agent LLM d'Enrichissement
| Principe MCP | Application |
|-------------|-------------|
| Transparence | Prompts visibles et modifiables, réponses explicites |
| Contrôle | Activation manuelle, pas d'automatisation sans validation |
| Isolation | Pas d'envoi de données PII, contexte limité aux données nécessaires |
| Auditabilité | Historique des prompts/réponses, templates versionnés |

## Recommandations pour une Architecture Responsable

1. **Minimisation des données** : Ne collecter que les données nécessaires à l'analyse
2. **Consentement éclairé** : L'utilisateur doit savoir exactement ce que chaque agent fait
3. **Humain dans la boucle** : Les décisions critiques doivent être validées par un humain
4. **Fail-safe** : En cas d'erreur, le système doit échouer de manière sécurisée
5. **Documentation** : Chaque composant doit être documenté avec ses responsabilités MCP
6. **Tests d'éthique** : Vérifier régulièrement que les agents respectent les principes MCP
7. **Mise à jour** : Suivre l'évolution de la spécification MCP (https://modelcontextprotocol.io/specification/2025-03-26)

## Conclusion

L'architecture MCP d'Anthropic fournit un cadre solide pour construire des agents IA
responsables et dignes de confiance. Notre projet Smart eCommerce Intelligence intègre
ces principes dès la conception :
- Agents transparents avec logs complets
- Contrôle utilisateur à chaque étape
- Isolation et sécurité des données
- Auditabilité complète du pipeline

Cette approche garantit que notre système reste éthique, contrôlable et bénéfique
pour les utilisateurs tout en exploitant la puissance des LLMs et du Machine Learning
pour l'intelligence économique.

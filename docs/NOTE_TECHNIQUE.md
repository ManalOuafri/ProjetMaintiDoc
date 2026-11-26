# MaintiDoc — Partie I : RAG multimodal agentique

**Note technique**

## 1. Objectif

Construire une application de question-réponse sur un PDF technique contenant
du texte, des images et des tableaux. Le contenu est indexé de façon multimodale
dans PostgreSQL/pgvector ; les réponses sont générées par un LLM open source
local (Qwen2.5 via Ollama), orchestré par un agent LangChain, et citent leurs
sources (document, page, type d'élément).

Document de démonstration : *Attention Is All You Need* (Vaswani et al., 2017),
11 pages, 3 tableaux et 2 figures (5 schémas), comme recommandé dans l'énoncé.

## 2. Architecture

```
INDEXATION
PDF ─► LlamaIndex PyMuPDFReader ─► texte/page ─► SentenceSplitter (512 tk, 64 chevauchement) ──┐
   ├─► PyMuPDF find_tables ─► tableaux (reconstruits ligne par ligne) ─────────────────────────┼─► BGE-M3 ─► data_maintidoc_text   (1024 d)
   └─► PyMuPDF images + figures vectorielles ─► PNG ─► Qwen2.5-VL 3B ─► description ───────────┘
                                                 └──► CLIP ViT-B/32 ─────────────────────────────────────► data_maintidoc_images (512 d)

QUESTION / RÉPONSE
Streamlit ─► Agent LangChain (ReAct, mémoire InMemorySaver)
               outils : search_documents · search_tables · search_figures · list_documents
               ─► recherche cosinus dans PGVector (filtre par type / document)
               ─► Qwen2.5 7B (Ollama) ─► réponse sourcée + passages, tableaux, figures
```

Toutes les briques tournent en local ; aucune donnée n'est envoyée à un service externe.

## 3. Modèles et outils retenus

| Rôle | Choix | Justification |
|---|---|---|
| Parsing PDF | LlamaIndex `PyMuPDFReader` + PyMuPDF | Local, rapide, conserve le numéro de page. LlamaParse (cloud) écarté pour rester 100 % local. |
| Découpage | `SentenceSplitter` 512 tokens / 64 de chevauchement | Respecte les phrases ; le chevauchement évite de couper une information entre deux chunks. |
| Embeddings texte | `BAAI/bge-m3` (1024 d) | Open source, multilingue : une question en français retrouve un passage en anglais. Modèle cité dans la feuille de route du projet. |
| Représentation des images | `openai/clip-vit-base-patch32` (512 d) | Espace commun texte/image : recherche d'une figure à partir d'une phrase. |
| Description des images | `qwen2.5vl:3b` (Ollama) | CLIP ne génère pas de texte. Le VLM décrit chaque figure ; la description est indexée avec BGE-M3, ce qui rend la figure interrogeable. Prépare l'étape 2 (Qwen2.5-VL). |
| Base vectorielle | PostgreSQL 16 + pgvector 0.8, index HNSW cosinus | Vecteurs et métadonnées (JSONB) dans la même base ; filtres SQL par type et document. |
| LLM | `qwen2.5:7b` (Ollama) | Open source, bon support du *tool calling*, tient sur la machine de test (RTX 3050 4 Go + CPU). |
| Agent | LangChain `create_agent` + LangGraph `InMemorySaver` | Boucle ReAct : le LLM choisit ses outils, peut relancer une recherche ; mémoire par conversation. |
| Interface | Streamlit | Chargement du PDF, conversation, affichage des sources, tableaux et figures. |

### Métadonnées stockées

Chaque vecteur est accompagné de : `doc_id` (hash du fichier), `doc_name`, `page`,
`type` (`text` / `table` / `image`), `caption` (légende « Table N » / « Figure N »)
et `image_path` pour les figures. Elles servent aux citations, aux filtres des
outils de l'agent, et au remplacement d'un document ré-indexé.

## 4. Fonctionnement

### 4.1 Indexation (`maintidoc/indexing.py`)

1. **Texte** : un `Document` LlamaIndex par page, découpé en chunks.
2. **Tableaux** : détection PyMuPDF limitée aux pages portant une légende « Table N »
   ou contenant beaucoup de traits. Le tableau est reconstruit ligne par ligne
   à partir de la position des mots (cellules séparées par `|`) : plus fiable que
   la grille de `find_tables` pour les tableaux « booktabs » sans traits entre lignes.
3. **Images** : images matricielles + figures vectorielles (amas de tracés
   `cluster_drawings` associés à une légende « Figure N »), rendues en PNG.
   Chaque image reçoit une description Qwen2.5-VL, puis deux vecteurs :
   BGE-M3 (description) et CLIP (pixels).

### 4.2 RAG classique (`maintidoc/rag.py`)

Pipeline fixe : 5 passages de texte + 2 tableaux + 2 descriptions de figures
+ 2 images CLIP → prompt (« réponds uniquement à partir du contexte, cite les
sources, sinon dis que tu ne sais pas ») → Qwen2.5 7B.

### 4.3 RAG agentique (`maintidoc/agent.py`)

L'agent reçoit la question et un prompt système, puis décide :
- quel(s) outil(s) appeler (texte, tableaux, figures, liste des documents) ;
- avec quelle requête (il peut reformuler, traduire en anglais pour CLIP) ;
- s'il faut relancer une recherche avant de répondre.

Les résultats des outils sont présentés comme des **données** (balises `<source>`),
et le prompt demande d'ignorer toute consigne qui y apparaîtrait (protection
élémentaire contre l'injection de prompt via les documents).

## 5. Résultats observés

### 5.1 Indexation du PDF de démonstration

| Élément | Résultat |
|---|---|
| Pages | 11 |
| Chunks de texte | 23 |
| Tableaux | 3 (Tables 1, 2, 3, avec leur légende) |
| Figures | 3 images (Figure 1 ; Figure 2 en deux schémas) |
| Vecteurs BGE-M3 (1024 d) | 29 (texte + tableaux + descriptions) |
| Vecteurs CLIP (512 d) | 3 |

### 5.2 Exemples de questions / réponses (agent, `qwen2.5:7b`)

Réponses reproduites telles que générées (`scripts/eval_questions.py`).

**1. Figure — « Que représente la figure 1 du document ? »**
Outil choisi : `search_figures("figure 1")`
> La figure 1 du document représente l'architecture du Transformer. C'est un diagramme
> qui illustre […] l'embedding d'entrée, l'encodage positionnel, l'attention multi-têtes,
> les couches feed-forward […] des couches de normalisation (Add & Norm) […] une couche
> linéaire et une fonction softmax pour produire des probabilités de sortie.
> [attention is all you need.pdf, p. 3, figure]

✅ Correct ; la réponse s'appuie sur la description générée par Qwen2.5-VL.

**2. Tableau — « Quel score BLEU obtient le Transformer (big) en anglais-allemand, et combien a coûté son entraînement ? »**
Outil choisi : `search_tables("score BLEU Transformer big anglais-allemand coût entraînement")`
> Le score BLEU pour le modèle Transformer (big) sur la traduction anglais-allemand est
> de 28.4 […]. Concernant le coût d'entraînement, il a nécessité environ 2.3 × 10¹⁹ FLOPs.

✅ Correct (Table 2, p. 8).

**3. Texte — « Pourquoi divise-t-on le produit scalaire par la racine carrée de dk ? »**
Outil choisi : `search_documents("pourquoi diviser produit scalaire racine carrée dk attention")`
> […] Cela permet d'éviter que les valeurs de produit scalaire grandissent en magnitude,
> ce qui pourrait entraîner des gradients très petits lors de l'utilisation de la
> fonction softmax […]. Référence : [attention is all you need.pdf, p. 4, texte]

✅ Correct. Lors d'un premier essai, l'agent avait répondu « je ne sais pas » **sans
appeler d'outil** ; un garde-fou relance désormais l'agent en imposant une recherche
quand aucun outil n'a été utilisé.

**4. Texte + tableau — « Combien de têtes d'attention utilise le modèle de base, et quelles sont les dimensions dk et dv ? »**
Outil choisi : `search_tables("têtes d'attention dimensions dk dv")`
> Le modèle de base utilise 8 têtes d'attention, avec des dimensions dk et dv de 64.
> [attention is all you need.pdf, p. 9, tableau 3]

✅ Correct.

**5. Hors corpus — « Quelle est la pression nominale de la pompe hydraulique P-200 ? »**
> Je ne sais pas : cette information n'est pas disponible dans les documents indexés.

✅ Pas d'hallucination.

**Comparaison avec le RAG classique (question 2)**
> Le Transformer (big) obtient un score BLEU de 28.4 […], et son entraînement a coûté
> environ 3.3 × 10¹⁸ FLOPs.

⚠️ BLEU correct, mais le coût cité est celui du modèle *base* : le contexte fixe
mélange plusieurs passages, et le LLM confond deux lignes voisines. L'agent, qui a
ciblé l'outil `search_tables`, donne la bonne valeur.

### 5.3 Temps de réponse (RTX 3050 4 Go, 16 Go RAM)

| Étape | Durée observée |
|---|---|
| Extraction PDF (texte + tableaux + figures) | ≈ 25 s |
| Description d'une figure (Qwen2.5-VL 3B) | 30–60 s (seul sur la machine) |
| Embeddings BGE-M3 via Ollama (GPU), 29 éléments | quelques secondes |
| Embeddings BGE-M3 via sentence-transformers (CPU) | ≈ 8 s par chunk |
| Réponse de l'agent | 3 à 8 min **pendant les tests**, Ollama étant partagé avec un autre projet (rechargements de modèles) |

Les temps de réponse de l'agent ont été mesurés dans de mauvaises conditions :
d'autres modèles occupaient Ollama, et la mémoire libre était faible. Ce sont des
résultats observés sur cette machine, pas des performances garanties.

## 6. Limites

1. **Matériel** : `qwen2.5:7b` ne tient pas entièrement dans 4 Go de VRAM (exécution
   partagée CPU/GPU) ; les réponses sont lentes, et l'indexation des figures aussi.
   Le modèle se change dans `.env` (`LLM_MODEL`) ou dans l'interface.
2. **Fiabilité de l'agent** : un LLM 7B ne choisit pas toujours d'appeler un outil
   (corrigé par un garde-fou), cite parfois ses sources de façon incomplète, et peut
   confondre deux lignes d'un tableau.
3. **Tableaux** : la détection PyMuPDF dépend des traits du tableau ; les tableaux
   sans aucun trait, ou scannés, ne sont pas détectés. Les exposants sont aplatis
   (`10^20` devient `1020` dans le texte extrait).
4. **Figures** : les figures vectorielles sont repérées grâce à leur légende
   « Figure N » ; une figure sans légende peut être ignorée. Les descriptions du VLM
   comportent des approximations (ex. « Q, K et V sont multipliés ensemble », alors
   que seuls Q et Kᵀ le sont).
5. **CLIP** : modèle entraîné sur des photos, surtout en anglais, et limité à 77
   tokens ; il distingue mal des schémas techniques entre eux (similarités ≈ 0,26–0,32).
   La recherche par description textuelle reste plus fiable ; l'agent combine les deux.
6. **PDF scannés** : pas d'OCR dans cette partie ; un PDF image ne produit pas de texte.
7. **Sécurité** : pas d'authentification ni de gestion des droits d'accès aux
   documents (prévu aux étapes 3 et 4) ; la protection contre l'injection de prompt
   se limite à une consigne dans le prompt système.
8. **Mémoire** : la mémoire de conversation (`InMemorySaver`) est perdue au
   redémarrage de l'application.

## 7. Pistes pour la suite

- Étape 2 : Qwen2.5-VL pour analyser une **photo** envoyée par le technicien, fusion
  des résultats texte/image et reranking (RRF + reranker multilingue).
- OCR local (Tesseract / docTR) pour les manuels scannés.
- Recherche hybride (BM25 + vecteurs) pour les références de pièces (« P-200 »),
  mal servies par la seule similarité sémantique.
- Corpus réel de maintenance (30 à 50 documents, deux familles d'équipements).

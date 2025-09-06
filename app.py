# streamlit web app for your chess prep
# run locally:  streamlit run app.py
import io, os, math, re, time, base64, datetime as dt
from typing import List, Tuple, Dict, Optional

import streamlit as st
import pandas as pd
import chess.pgn
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Chess Prep", layout="wide")

# ------------------ utils PGN & ECO ------------------
def parse_pgn_stream(pgn_text: str):
    stream = io.StringIO(pgn_text)
    while True:
        g = chess.pgn.read_game(stream)
        if g is None: break
        yield g

def extract_opening_info(g: chess.pgn.Game) -> Tuple[str, str]:
    eco = (g.headers.get("ECO") or "").strip()
    name = (g.headers.get("Opening") or "").strip()
    return eco, name

def first_line_SAN(g: chess.pgn.Game, max_plies: int) -> str:
    bd = g.board(); node = g; plies = 0; moves = []
    while node.variations and plies < max_plies:
        node = node.variations[0]
        moves.append(bd.san(node.move))
        bd.push(node.move); plies += 1
    return " ".join(moves)

def stats_from_pgn_fixed_side(pgn_text: str, side: str, max_plies: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_open, rows_line = [], []
    side = side.lower()
    for g in parse_pgn_stream(pgn_text):
        res_tag = (g.headers.get("Result") or "")
        if res_tag not in ("1-0","0-1","1/2-1/2","½-½"): continue
        if side == "white":
            res = 1.0 if res_tag=="1-0" else 0.5 if ("1/2" in res_tag or "½" in res_tag) else 0.0
            my_elo  = int((g.headers.get("WhiteElo") or "0") or "0")
            opp_elo = int((g.headers.get("BlackElo") or "0") or "0")
        else:
            res = 1.0 if res_tag=="0-1" else 0.5 if ("1/2" in res_tag or "½" in res_tag) else 0.0
            my_elo  = int((g.headers.get("BlackElo") or "0") or "0")
            opp_elo = int((g.headers.get("WhiteElo") or "0") or "0")
        eco, opening = extract_opening_info(g)
        line = first_line_SAN(g, max_plies)
        rows_open.append({"eco":eco,"opening":opening,"games":1,"score_sum":res,
                          "win":1 if res==1 else 0,"draw":1 if res==0.5 else 0,"loss":1 if res==0 else 0,
                          "opp_elo_sum":opp_elo,"my_elo_sum":my_elo})
        rows_line.append({"eco":eco,"opening":opening,"line":line,"games":1,"score_sum":res,
                          "win":1 if res==1 else 0,"draw":1 if res==0.5 else 0,"loss":1 if res==0 else 0,
                          "opp_elo_sum":opp_elo,"my_elo_sum":my_elo})
    def agg(rows, add_line=False):
        if not rows:
            cols=["eco","opening","games","score_pct","win","draw","loss","avg_opp_elo","avg_my_elo"]
            if add_line: cols.insert(2,"line")
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        group_cols = ["eco","opening"] + (["line"] if add_line else [])
        df = df.groupby(group_cols, as_index=False).sum(numeric_only=True)
        df["score_pct"] = (100.0*df["score_sum"]/df["games"]).round(1)
        df["avg_opp_elo"] = (df["opp_elo_sum"]/df["games"]).round(0).astype(int)
        df["avg_my_elo"]  = (df["my_elo_sum"]/df["games"]).round(0).astype(int)
        df = df.drop(columns=["score_sum","opp_elo_sum","my_elo_sum"])
        df = df.sort_values(["score_pct","games"], ascending=[False,False], ignore_index=True)
        return df
    return agg(rows_open, False), agg(rows_line, True)

def compare_cross_colors(
    opp_df_open, me_df_open, opp_df_line, me_df_line,
    min_games_opening=5, min_games_line=3,
    min_games_me_opening=8, min_games_me_line=5,
    restrict_to_positive_openings=True, weighted_sort=True
):
    mo = pd.merge(opp_df_open, me_df_open, on=["eco","opening"], suffixes=("_opp","_me"))
    mo = mo[(mo["games_opp"] >= min_games_opening) & (mo["games_me"] >= min_games_me_opening)].copy()
    mo["score_gap"] = (mo["score_pct_me"] - mo["score_pct_opp"]).round(1)
    if weighted_sort:
        mo["_w"] = mo["games_opp"].apply(lambda x: math.log1p(x)) * mo["games_me"].apply(lambda x: math.log1p(x))
        mo["_rank"] = mo["score_gap"] * mo["_w"]
        mo = mo.sort_values(["_rank","games_opp","games_me"], ascending=[False,False,False], ignore_index=True).drop(columns=["_w","_rank"])
    else:
        mo = mo.sort_values(["score_gap","games_opp","games_me"], ascending=[False,False,False], ignore_index=True)

    ml = pd.merge(opp_df_line, me_df_line, on=["eco","opening","line"], suffixes=("_opp","_me"))
    ml = ml[(ml["games_opp"] >= min_games_line) & (ml["games_me"] >= min_games_me_line)].copy()
    if restrict_to_positive_openings and not mo.empty:
        good_keys = mo.loc[mo["score_gap"] > 0, ["eco","opening"]].drop_duplicates()
        ml = ml.merge(good_keys, on=["eco","opening"], how="inner")
    ml["score_gap"] = (ml["score_pct_me"] - ml["score_pct_opp"]).round(1)
    if weighted_sort:
        ml["_w"] = ml["games_opp"].apply(lambda x: math.log1p(x)) * ml["games_me"].apply(lambda x: math.log1p(x))
        ml["_rank"] = ml["score_gap"] * ml["_w"]
        ml = ml.sort_values(["_rank","games_opp","games_me"], ascending=[False,False,False], ignore_index=True).drop(columns=["_w","_rank"])
    else:
        ml = ml.sort_values(["score_gap","games_opp","games_me"], ascending=[False,False,False], ignore_index=True)
    return mo, ml

def autotune_lines_to_target(opp_open, me_open, opp_line, me_line, params: dict, target: int):
    eff = params.copy()
    mo, ml = compare_cross_colors(opp_open, me_open, opp_line, me_line, **eff)
    if len(ml) >= target:
        return mo, ml, eff, False
    tuned = False
    while len(ml) < target and eff["min_games_me_line"] > 1:
        eff["min_games_me_line"] -= 1
        mo, ml = compare_cross_colors(opp_open, me_open, opp_line, me_line, **eff); tuned = True
    while len(ml) < target and eff["min_games_line"] > 1:
        eff["min_games_line"] -= 1
        mo, ml = compare_cross_colors(opp_open, me_open, opp_line, me_line, **eff); tuned = True
    if len(ml) < target and eff.get("restrict_to_positive_openings", True):
        eff["restrict_to_positive_openings"] = False
        mo, ml = compare_cross_colors(opp_open, me_open, opp_line, me_line, **eff); tuned = True
    return mo, ml, eff, tuned

def eco_is_open_game(eco: str) -> bool:
    try:
        eco = (eco or "").upper()
        return (eco.startswith("C") and 20 <= int(eco[1:]) <= 59)
    except Exception:
        return False

def pick_unique_reco(mo: pd.DataFrame, ml: pd.DataFrame,
                     prefer_open_games: bool = False,
                     exclude_irregular: bool = False,
                     min_opp_games_reco: Optional[int] = None):
    if mo.empty:
        return None, None
    cand = mo.copy()
    if isinstance(min_opp_games_reco, int):
        tmp = cand[cand["games_opp"] >= int(min_opp_games_reco)]
        if not tmp.empty: cand = tmp
    if exclude_irregular:
        tmp = cand[~cand["eco"].isin(["A00","A01"])]
        if not tmp.empty: cand = tmp
    if prefer_open_games:
        scotch = cand[cand["opening"].str.contains("Scotch", na=False)]
        if not scotch.empty: cand = scotch
        else:
            og = cand[cand["eco"].apply(eco_is_open_game)]
            if not og.empty: cand = og
    o = cand.iloc[0].to_dict()
    lines = ml[(ml["eco"]==o.get("eco")) & (ml["opening"]==o.get("opening"))]
    l = lines.iloc[0].to_dict() if not lines.empty else None
    return o, l

# deep line helpers
def _tokenize_san(line: Optional[str]) -> List[str]:
    if not line: return []
    line = re.sub(r"\d+\.(\.\.)?\s*", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    return [t for t in line.split(" ") if t]

def _first_n_san(g: chess.pgn.Game, nplies: int) -> List[str]:
    bd = g.board(); node = g; plies = 0; seq = []
    while node.variations and plies < nplies:
        node = node.variations[0]
        seq.append(bd.san(node.move)); bd.push(node.move); plies += 1
    return seq

def _starts_with_prefix(seq: List[str], prefix: List[str]) -> bool:
    if not prefix: return True
    if len(seq) < len(prefix): return False
    return all(seq[i]==mv for i, mv in enumerate(prefix))

def find_deep_line(pgn_text: str, my_side: str, eco: Optional[str], opening_name: Optional[str],
                   prefix_line: Optional[str] = None, deep_plies: int = 20) -> Tuple[Optional[str], int, Optional[float]]:
    my_side = (my_side or "white").lower()
    prefix = _tokenize_san(prefix_line)
    line_stats: Dict[Tuple[str,...], Dict[str,float]] = {}
    for g in parse_pgn_stream(pgn_text):
        g_eco, g_name = extract_opening_info(g)
        ok = False
        if eco and g_eco: ok = (g_eco == eco)
        if not ok and opening_name: ok = opening_name.lower() in (g_name or "").lower()
        if not ok: continue
        seq = _first_n_san(g, deep_plies)
        if not _starts_with_prefix(seq, prefix): continue
        res = g.headers.get("Result","")
        if my_side=="white":
            s = 1.0 if res=="1-0" else 0.5 if ("1/2" in res or "½" in res) else 0.0
        else:
            s = 1.0 if res=="0-1" else 0.5 if ("1/2" in res or "½" in res) else 0.0
        key = tuple(seq)
        if not key: continue
        d = line_stats.get(key, {"games":0, "score_sum":0.0})
        d["games"] += 1; d["score_sum"] += s
        line_stats[key] = d
    if not line_stats and prefix:
        return find_deep_line(pgn_text, my_side, eco, opening_name, prefix_line=None, deep_plies=deep_plies)
    if not line_stats: return None, 0, None
    best_key, best_metric, best_games, best_score = None, -1.0, 0, 0.0
    for key, d in line_stats.items():
        games = int(d["games"]); score_pct = 100.0 * d["score_sum"] / games
        metric = score_pct * math.log1p(games)
        if metric > best_metric:
            best_metric = metric; best_key = key; best_games = games; best_score = score_pct
    return " ".join(best_key) if best_key else None, best_games, round(best_score,1)

def san_to_pgn_numbered(san_line: str, start_move_no: int = 1) -> str:
    if not san_line: return ""
    toks = _tokenize_san(san_line); out=[]; mv=start_move_no
    for i in range(0,len(toks),2):
        out.append(f"{mv}. {toks[i]}" + (f" {toks[i+1]}" if i+1 < len(toks) else ""))
        mv += 1
    return " ".join(out)

# --------- chess.com profiles ----------
HTTP_HEADERS = {"User-Agent": "chess-prep-web/1.0"}
def http_get_json(url: str, timeout=15):
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        return r.json() if r.status_code==200 else None
    except Exception:
        return None

def fetch_archives(username: str) -> List[str]:
    data = http_get_json(f"https://api.chess.com/pub/player/{username}/games/archives")
    return list(data.get("archives", [])) if data else []

def fetch_archive_games(url: str, username: str) -> List[Dict]:
    data = http_get_json(url); out=[]
    if not data: return out
    for g in data.get("games", []):
        end_ts = g.get("end_time"); 
        if not end_ts: continue
        rating = None
        if g.get("white",{}).get("username","").lower()==username.lower():
            rating = g.get("white",{}).get("rating")
        elif g.get("black",{}).get("username","").lower()==username.lower():
            rating = g.get("black",{}).get("rating")
        if rating:
            out.append({"end": dt.datetime.utcfromtimestamp(int(end_ts)),
                        "rating": int(rating),
                        "time_class": (g.get("time_class") or "").lower()})
    return out

def monthly_median(series: List[Dict]) -> pd.DataFrame:
    if not series: return pd.DataFrame(columns=["time_class","month","rating"])
    df = pd.DataFrame(series); df["month"]=df["end"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["time_class","month"],as_index=False)["rating"].median().sort_values(["time_class","month"])

def pick_current_ratings(stats: Dict) -> Dict[str, Optional[int]]:
    out={}
    for k_api,k_out in [("chess_bullet","bullet"),("chess_blitz","blitz"),("chess_rapid","rapid"),("chess_daily","daily")]:
        node=(stats or {}).get(k_api) or {}
        last=(node.get("last") or {}).get("rating")
        out[k_out]= int(last) if isinstance(last,(int,float)) else None
    return out

def fetch_player_profile(username: str) -> Optional[Dict]:
    return http_get_json(f"https://api.chess.com/pub/player/{username}")

def fetch_player_stats(username: str) -> Optional[Dict]:
    return http_get_json(f"https://api.chess.com/pub/player/{username}/stats")

def elo_expected(pA: int, pB: int) -> float:
    return 1.0/(1.0+10**((pB-pA)/400.0))

# --------------- UI --------------------
st.title("Préparation d’adversaire — version Web")

tab_an, tab_prof = st.tabs(["🔍 Analyse & reco", "📈 Profils & Elo"])

with tab_an:
    st.sidebar.header("Paramètres")
    prefer_open = st.sidebar.checkbox("Favoriser les jeux ouverts (1.e4 e5)", value=True)
    exclude_irreg = st.sidebar.checkbox("Écarter A00 / A01 (irrégulières)", value=True)
    min_opp_reco = st.sidebar.number_input("Seuil min parties adv (reco)", 1, 100, 10)
    max_plies = st.sidebar.slider("Ligne courte : demi-coups", 6, 20, 10)
    min_adv_open = st.sidebar.number_input("Min parties adv — ouverture", 1, 100, 5)
    min_adv_line  = st.sidebar.number_input("Min parties adv — ligne", 1, 100, 3)
    min_me_open   = st.sidebar.number_input("Min parties moi — ouverture", 1, 100, 8)
    min_me_line   = st.sidebar.number_input("Min parties moi — ligne", 1, 100, 5)
    autotune = st.sidebar.checkbox("Auto-paramétrage : assurer ≥ N lignes", value=True)
    target_lines = st.sidebar.number_input("N lignes visées", 1, 15, 3)

    st.subheader("Entrées")
    col = st.columns(2)
    with col[0]:
        p1 = st.text_input("Joueur 1 (username Chess.com)", "yannbernard76")
        p1w = st.file_uploader("PGN P1 — Blancs", type=["pgn"])
        p1b = st.file_uploader("PGN P1 — Noirs",  type=["pgn"])
    with col[1]:
        p2 = st.text_input("Joueur 2 (username Chess.com)", "nalou14")
        p2w = st.file_uploader("PGN P2 — Blancs", type=["pgn"])
        p2b = st.file_uploader("PGN P2 — Noirs",  type=["pgn"])

    who_white = st.radio("Couleurs du match", ["P1 a les Blancs", "P2 a les Blancs"], index=0)
    preparing = st.radio("Qui se prépare ?", ["P1","P2"], index=0)

    run = st.button("Lancer l'analyse")

    if run:
        if not all([p1w, p1b, p2w, p2b]):
            st.error("Merci d’ajouter les 4 PGN (P1/P2, Blancs/Noirs).")
        else:
            # lire les PGN uploadés
            get = lambda f: f.getvalue().decode("utf-8", errors="ignore")
            p1w_txt, p1b_txt, p2w_txt, p2b_txt = get(p1w), get(p1b), get(p2w), get(p2b)

            # stats par couleur
            p1_open_W, p1_line_W = stats_from_pgn_fixed_side(p1w_txt, "white", max_plies)
            p1_open_B, p1_line_B = stats_from_pgn_fixed_side(p1b_txt, "black", max_plies)
            p2_open_W, p2_line_W = stats_from_pgn_fixed_side(p2w_txt, "white", max_plies)
            p2_open_B, p2_line_B = stats_from_pgn_fixed_side(p2b_txt, "black", max_plies)

            if preparing=="P1":
                me_name, opp_name = p1, p2
                if who_white.startswith("P1"):
                    opp_open, opp_line = p2_open_B, p2_line_B
                    me_open, me_line = p1_open_W, p1_line_W
                    me_side="white"; white_name, black_name = p1, p2
                    me_pgn_for_deep = p1w_txt
                else:
                    opp_open, opp_line = p2_open_W, p2_line_W
                    me_open, me_line = p1_open_B, p1_line_B
                    me_side="black"; white_name, black_name = p2, p1
                    me_pgn_for_deep = p1b_txt
            else:
                me_name, opp_name = p2, p1
                if who_white.startswith("P2"):
                    opp_open, opp_line = p1_open_B, p1_line_B
                    me_open, me_line = p2_open_W, p2_line_W
                    me_side="white"; white_name, black_name = p2, p1
                    me_pgn_for_deep = p2w_txt
                else:
                    opp_open, opp_line = p1_open_W, p1_line_W
                    me_open, me_line = p2_open_B, p2_line_B
                    me_side="black"; white_name, black_name = p1, p2
                    me_pgn_for_deep = p2b_txt

            params = dict(
                min_games_opening=int(min_adv_open),
                min_games_line=int(min_adv_line),
                min_games_me_opening=int(min_me_open),
                min_games_me_line=int(min_me_line),
                restrict_to_positive_openings=True,
                weighted_sort=True
            )

            mo, ml = compare_cross_colors(opp_open, me_open, opp_line, me_line, **params)
            tuned=False; eff=params.copy()
            if autotune and len(ml)<target_lines:
                mo, ml, eff, tuned = autotune_lines_to_target(opp_open, me_open, opp_line, me_line, params, int(target_lines))

            st.subheader("Ouvertures à cibler")
            st.dataframe(mo.head(50))
            st.subheader("Lignes courtes à cibler")
            st.dataframe(ml.head(50))

            # reco
            reco_opening, reco_line = pick_unique_reco(
                mo, ml,
                prefer_open_games=prefer_open,
                exclude_irregular=exclude_irreg,
                min_opp_games_reco=int(min_opp_reco)
            )

            if not reco_opening:
                st.warning("Pas de reco trouvée avec ces filtres.")
            else:
                eco = reco_opening.get("eco","")
                opening_name = reco_opening.get("opening","")
                short_line = (reco_line or {}).get("line")
                pv_line, deep_games, deep_score = find_deep_line(
                    me_pgn_for_deep, me_side, eco, opening_name, prefix_line=short_line, deep_plies=20
                )
                pv = san_to_pgn_numbered(pv_line) if pv_line else None

                st.markdown("### ✅ Recommandation unique")
                st.write(f"**{eco} — {opening_name}**")
                st.write(f"Avantage (score_gap): **{reco_opening.get('score_gap')} pts** • Volumes (adv/moi): {reco_opening.get('games_opp')} / {reco_opening.get('games_me')}")
                if short_line: st.write(f"Ligne courte: `{short_line}`")
                if pv: st.write(f"Ligne poussée (~20 plies): `{pv}`")

                # fichier PGN prêt à télécharger
                pgn_text = f"""[Event "Préparation web"]
[Site "?"]
[White "{white_name}"]
[Black "{black_name}"]
[Result "*"]
[ECO "{eco}"]
[Opening "{opening_name}"]

{pv if pv else '; indisponible'}
*"""
                st.download_button("📥 Télécharger le PGN de la reco", pgn_text.encode("utf-8"),
                                   file_name="recommendation.pgn", mime="application/x-chess-pgn")


                # push lichess (optionnel) — version nettoyée
                def _get_secret(name: str, default: str = ""):
                    # lit d'abord dans st.secrets (Cloud), sinon dans les variables d'environnement (local)
                    try:
                        import streamlit as st  # déjà importé plus haut
                        val = st.secrets.get(name)
                        if val:
                            return str(val)
                    except Exception:
                        pass
                    import os
                    return os.environ.get(name, default)

                with st.expander("Envoyer vers une étude Lichess (optionnel)"):
                    # NE PAS préremplir : l’utilisateur peut coller manuellement si besoin.
                    token_input = st.text_input(
                        "Token API Lichess (secret)",
                        type="password",
                        placeholder="colle le token ici si tu ne l'as pas mis dans les Secrets"
                    )
                    study_input = st.text_input(
                        "Study ID (optionnel)",
                        placeholder="ex: yfCjZd8R"
                    )

                    if st.button("Envoyer"):
                        try:
                            # Si rien saisi, on tente de lire les secrets/variables d'env
                            token = token_input or _get_secret("LICHESS_TOKEN", "")
                            study_id = study_input or _get_secret("LICHESS_STUDY_ID", "")

                            # Construire les en-têtes uniquement si on a un token
                            headers = {"Authorization": f"Bearer {token}"} if token else {}

                            if token and study_id:
                                # Importer le PGN en tant que NOUVEAU chapitre d'une étude
                                r = requests.post(
                                    f"https://lichess.org/api/study/{study_id}/import-pgn",
                                    headers=headers,
                                    data={"pgn": pgn_text, "chapterName": f"Prépa – {eco} {opening_name}"},
                                    timeout=30
                                )
                                r.raise_for_status()
                                st.success("Importé dans l'étude. Ouvre l'étude pour voir le nouveau chapitre.")
                            else:
                                # Import simple (pas besoin de token ni d'ID d'étude)
                                r = requests.post("https://lichess.org/api/import", data={"pgn": pgn_text}, timeout=30)
                                r.raise_for_status()
                                url = r.json().get("url")
                                st.success(f"Partie importée : {url}")
                                st.markdown(f"[Ouvrir sur Lichess]({url})")
                        except Exception as e:
                            st.error(f"Échec envoi Lichess : {e}")

with tab_prof:
    st.subheader("Profils Chess.com (Elo & volumes)")
    c1, c2 = st.columns(2)
    with c1:
        u1 = st.text_input("Joueur A (username Chess.com)", "yannbernard76", key="pa")
    with c2:
        u2 = st.text_input("Joueur B (username Chess.com)", "nalou14", key="pb")
    go = st.button("Récupérer")

    if go:
        for uname in (u1, u2):
            if not uname: st.error("Renseigne les deux usernames."); st.stop()

        # profils + stats actuelles
        stats1, stats2 = fetch_player_stats(u1) or {}, fetch_player_stats(u2) or {}
        cur1, cur2 = pick_current_ratings(stats1), pick_current_ratings(stats2)

        st.markdown("#### Élos actuels")
        df_cur = pd.DataFrame({
            "cadence":["bullet","blitz","rapid","daily"],
            u1:[cur1.get("bullet"),cur1.get("blitz"),cur1.get("rapid"),cur1.get("daily")],
            u2:[cur2.get("bullet"),cur2.get("blitz"),cur2.get("rapid"),cur2.get("daily")],
        })
        st.dataframe(df_cur)

        # archives → séries
        series = {}
        for uname in (u1, u2):
            entries=[]
            for aurl in fetch_archives(uname):
                time.sleep(0.1)
                entries.extend(fetch_archive_games(aurl, uname))
            series[uname]=entries

        st.markdown("#### Évolution Elo (médiane mensuelle)")
        for tc in ["blitz","rapid","bullet"]:
            fig, ax = plt.subplots(figsize=(7,3))
            for uname, color in [(u1, None),(u2, None)]:
                df = monthly_median(series[uname])
                d = df[df["time_class"]==tc]
                if len(d): ax.plot(d["month"], d["rating"], label=f"{uname}")
            ax.set_title(tc); ax.grid(True, alpha=.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

        st.markdown("#### Proba Elo (si on suppose blitz)")
        if cur1.get("blitz") and cur2.get("blitz"):
            p = elo_expected(cur1["blitz"], cur2["blitz"])
            st.write(f"P({u1} bat {u2}) ≈ **{p*100:.1f}%** • ΔElo = {cur1['blitz']-cur2['blitz']}")
        else:
            st.write("_Indisponible_")

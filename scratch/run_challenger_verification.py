import os
import sys
import re
from pathlib import Path
from PIL import Image
import pandas as pd

# Add src/polymarket_weather to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "polymarket_weather"))

import evaluate_oos

def verify_report_file(report_path):
    print("--- 1. Checking meta_analysis_report.md ---")
    p = Path(report_path)
    if not p.exists():
        return False, f"Report file {report_path} does not exist."
    size = p.stat().st_size
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    if size == 0 or len(lines) == 0:
        return False, f"Report file is empty (size={size}, lines={len(lines)})."
    print(f"✅ Report exists, size={size} bytes, {len(lines)} lines.")
    return True, f"Report valid ({size} bytes, {len(lines)} lines)"

def verify_png_images(scratch_dir, image_names):
    print("\n--- 2. Checking PNG image artifacts ---")
    results = {}
    all_valid = True
    for img_name in image_names:
        img_path = Path(scratch_dir) / img_name
        if not img_path.exists():
            print(f"❌ {img_name}: File does not exist")
            results[img_name] = (False, "File does not exist")
            all_valid = False
            continue
        size_bytes = img_path.stat().st_size
        if size_bytes == 0:
            print(f"❌ {img_name}: File size is 0 bytes")
            results[img_name] = (False, "Size is 0 bytes")
            all_valid = False
            continue
        try:
            with Image.open(img_path) as img:
                fmt = img.format
                width, height = img.size
                img.verify()
            if fmt != 'PNG':
                print(f"❌ {img_name}: Format is {fmt}, expected PNG")
                results[img_name] = (False, f"Format is {fmt}")
                all_valid = False
            elif width <= 0 or height <= 0:
                print(f"❌ {img_name}: Invalid dimensions {width}x{height}")
                results[img_name] = (False, f"Invalid dimensions {width}x{height}")
                all_valid = False
            else:
                print(f"✅ {img_name}: PNG valid, size={size_bytes} bytes, dimensions={width}x{height}")
                results[img_name] = (True, f"Valid PNG, {size_bytes} bytes, {width}x{height}")
        except Exception as e:
            print(f"❌ {img_name}: Exception loading image: {e}")
            results[img_name] = (False, f"Corrupt image: {e}")
            all_valid = False
    return all_valid, results

def verify_evaluate_oos_metrics(report_path):
    print("\n--- 3. Verifying Brier scores & Log-Loss calculations ---")
    out_dir = Path(__file__).resolve().parent.parent / "src" / "polymarket_weather" / "output"
    calibrated_path = out_dir / "opportunities_evaluation_calibrated.csv"
    ensemble_path = out_dir / "opportunities_evaluation_ensemble.csv"
    
    ml = evaluate_oos._graded_markets(calibrated_path)
    if ml is None or ml.empty:
        return False, "Failed to load graded markets from calibrated tracker"
    
    y = ml["outcome"].tolist()
    p_model = ml["forecast_prob"].tolist()
    mkt_col = "market_prob_raw" if "market_prob_raw" in ml.columns else "market_prob"
    p_mkt = ml[mkt_col].tolist()
    
    calc_brier_model = evaluate_oos._brier(p_model, y)
    calc_brier_mkt = evaluate_oos._brier(p_mkt, y)
    calc_logloss_model = evaluate_oos._logloss(p_model, y)
    calc_logloss_mkt = evaluate_oos._logloss(p_mkt, y)
    
    ens = evaluate_oos._graded_markets(ensemble_path)
    common = set(ml["condition_id"]) & set(ens["condition_id"])
    e = ens[ens["condition_id"].isin(common)].set_index("condition_id")
    m = ml[ml["condition_id"].isin(common)].set_index("condition_id")
    ye = e["outcome"].tolist()
    calc_brier_ens = evaluate_oos._brier(e["forecast_prob"].tolist(), ye)
    calc_logloss_ens = evaluate_oos._logloss(e["forecast_prob"].tolist(), ye)
    calc_brier_model_paired = evaluate_oos._brier(m["forecast_prob"].tolist(), m["outcome"].tolist())
    
    crps_m_by = evaluate_oos._crps_by_key(calibrated_path)
    crps_e_by = evaluate_oos._crps_by_key(ensemble_path)
    common_crps = set(crps_m_by) & set(crps_e_by)
    calc_crps_m_paired = sum(crps_m_by[k] for k in common_crps) / len(common_crps)
    calc_crps_e_paired = sum(crps_e_by[k] for k in common_crps) / len(common_crps)
    calc_crps_delta = calc_crps_m_paired - calc_crps_e_paired

    city_briers = {}
    for city, grp in ml.groupby("city"):
        yc = grp["outcome"].tolist()
        city_briers[str(city)] = {
            "n": len(grp),
            "model_brier": evaluate_oos._brier(grp["forecast_prob"].tolist(), yc),
            "market_brier": evaluate_oos._brier(grp[mkt_col].tolist(), yc)
        }

    report_text = Path(report_path).read_text(encoding="utf-8")
    
    checks = []
    
    # 1. Overall Brier Model
    m_brier_match = f"{calc_brier_model:.4f}" in report_text
    checks.append(("Overall Model Brier (0.1546)", m_brier_match, f"Calculated: {calc_brier_model:.4f}"))
    
    # 2. Overall Brier Market
    mkt_brier_match = f"{calc_brier_mkt:.4f}" in report_text
    checks.append(("Overall Market Brier (0.1213)", mkt_brier_match, f"Calculated: {calc_brier_mkt:.4f}"))

    # 3. Overall Log-Loss Model
    m_ll_match = f"{calc_logloss_model:.4f}" in report_text
    checks.append(("Overall Model Log-Loss (0.5238)", m_ll_match, f"Calculated: {calc_logloss_model:.4f}"))

    # 4. Overall Log-Loss Market
    mkt_ll_match = f"{calc_logloss_mkt:.4f}" in report_text
    checks.append(("Overall Market Log-Loss (0.3846)", mkt_ll_match, f"Calculated: {calc_logloss_mkt:.4f}"))

    # 5. Paired Brier Ensemble (0.1502)
    ens_brier_match = f"{calc_brier_ens:.4f}" in report_text
    checks.append(("Paired Ensemble Brier (0.1502)", ens_brier_match, f"Calculated: {calc_brier_ens:.4f}"))

    # 6. Paired Brier Model (0.1556)
    model_paired_match = f"{calc_brier_model_paired:.4f}" in report_text
    checks.append(("Paired Model Brier (0.1556)", model_paired_match, f"Calculated: {calc_brier_model_paired:.4f}"))

    # 7. Paired Log-Loss Ensemble (0.5707)
    ens_ll_match = f"{calc_logloss_ens:.4f}" in report_text
    checks.append(("Paired Ensemble Log-Loss (0.5707)", ens_ll_match, f"Calculated: {calc_logloss_ens:.4f}"))

    # 8. Paired CRPS Model (1.3384)
    crps_m_match = f"{calc_crps_m_paired:.4f}" in report_text
    checks.append(("Paired Model CRPS (1.3384)", crps_m_match, f"Calculated: {calc_crps_m_paired:.4f}"))

    # 9. Paired CRPS Ensemble (1.2790)
    crps_e_match = f"{calc_crps_e_paired:.4f}" in report_text
    checks.append(("Paired Ensemble CRPS (1.2790)", crps_e_match, f"Calculated: {calc_crps_e_paired:.4f}"))

    # 10. City Briers
    for city, cb in city_briers.items():
        mb_str = f"{cb['model_brier']:.4f}"
        mkb_str = f"{cb['market_brier']:.4f}"
        city_match = mb_str in report_text and mkb_str in report_text
        checks.append((f"City Brier {city} (Model: {mb_str}, Market: {mkb_str})", city_match, f"Calculated Model={mb_str}, Market={mkb_str}"))

    all_passed = True
    for desc, status, detail in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {desc}: {detail}")
        if not status:
            all_passed = False
            
    return all_passed, checks

def main():
    scratch_dir = Path(__file__).resolve().parent
    report_path = scratch_dir / "meta_analysis_report.md"
    images = [
        "win_rate_by_price_bucket.png",
        "brier_score_comparison.png",
        "probability_gap_brier_deficit.png",
        "reliability_diagram.png"
    ]
    
    rep_ok, rep_msg = verify_report_file(report_path)
    img_ok, img_results = verify_png_images(scratch_dir, images)
    calc_ok, calc_checks = verify_evaluate_oos_metrics(report_path)
    
    overall = rep_ok and img_ok and calc_ok
    print(f"\n==================================================")
    print(f"VERDICT: {'PASS' if overall else 'FAIL'}")
    print(f"==================================================")
    return 0 if overall else 1

if __name__ == "__main__":
    sys.exit(main())

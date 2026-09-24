#!/bin/bash
# 双击运行：清理所有日志和生成结果 -> 全部重跑 -> 与上一次提交对比 -> git commit
cd "$(dirname "$0")"
rm -f .git/HEAD.lock .git/index.lock .git/objects/maintenance.lock .git/objects/*/tmp_obj_* 2>/dev/null
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
command -v conda >/dev/null || { echo "找不到 conda"; read -r -p "按回车关闭"; exit 1; }

echo ">>> 清理日志和生成结果"
rm -rf runs run_all_*.log run_summary.txt REPRODUCIBILITY.md \
  ho_pdf_revision/logs ho_pdf_revision/results \
  lv_gls_bdf2_vtk_and_code/{diagnostics,logs,vtk,monitor_gls.csv,run_metadata.json,short_control_comparison.json,validation_summary.json} \
  lv_lagrange_vtk_and_code/{diagnostics,logs,vtk,monitor_lagrange_full_cycle.csv,pure_vs_augmented.json,run_metadata.json,validation_summary.json} \
  p1_plain_full_cycle_program/{reference_output,output} \
  {ho_pdf_revision,lv_gls_bdf2_vtk_and_code,lv_lagrange_vtk_and_code}/source/reference_output
find . -name __pycache__ -type d -prune -exec rm -rf {} +

LOG="rerun_$(date +%Y%m%d_%H%M%S).log"
bash run_all.sh "${LV_ENV:-lv0d3d}" 2>&1 | tee "$LOG"
python tools/repro_tools.py compare REPRODUCIBILITY.md run_summary.txt
git add -A
git commit -q -F - <<'MSG'
Clean all logs/results and regenerate them from scratch on macOS

run_all.sh now reproduces every result of the package in its original
location (P1, Lagrange augmented + pure, original GLS full/smoke/no-stab,
HO-PDF GLS smoke/smoke_fs/full + verification). REPRODUCIBILITY.md compares
the regenerated CSV/JSON files with the previous commit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vBaRkHkRevJaxvWbQMZwb
MSG
git log --oneline -1
read -r -p "完成，按回车关闭窗口"

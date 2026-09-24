#!/bin/bash
# 双击运行：清理 git 残留锁文件，然后用 miniconda 运行三个程序
cd "$(dirname "$0")"
rm -f .git/HEAD.lock .git/objects/maintenance.lock .git/objects/*/tmp_obj_* 2>/dev/null
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
command -v conda >/dev/null || { echo "找不到 conda"; read -r -p "按回车关闭"; exit 1; }
bash run_all.sh "${LV_ENV:-lv0d3d}" 2>&1 | tee "run_all_$(date +%Y%m%d_%H%M%S).log"
read -r -p "完成，按回车关闭窗口"

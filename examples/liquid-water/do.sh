
for dir in ??; 
do 
  pushd $dir
  bash do.sh
  popd
done

for dir in ?? 
do 
  echo -n "${dir} " 
  tail -1 $dir/rmse.dat 
done | awk '{print $1, $4}' > rmses.dat

for dir in ??
do 
  echo -n "${dir} "
  tail -1 $dir/learning*
done > learning-last.dat 


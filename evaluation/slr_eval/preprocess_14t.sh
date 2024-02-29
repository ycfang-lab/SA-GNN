#!/bin/bash

if [ -z "$2" ];then
echo "preprocess_14t.sh <hypothesis-CTM-file> <output-file>"
exit 0
fi

hypothesisCTM=$1
output=$2

# apply some simplifications to the recognition
echo "preprocess_14t.sh ${hypothesisCTM} ${output}"
cat ${hypothesisCTM} | grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" | grep -v -e ' __[^_ ]*__$'| sed -e 's, loc-\([^ ]*\)$, \1,g' -e 's,-PLUSPLUS$,,g' -e 's, cl-\([^ ]*\)$, \1,g' -e 's,\b\([A-Z][A-Z]*\)RAUM$,\1,g' -e 's,\s*$,,'| awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' | awk 'BEGIN{prec=="";precID=""}{if (($NF!=prec)||($1!=precID)){print $0}precID=$1; prec=$NF}'|sort -k1,1 -k3,3  > ${output}


echo `date`
echo "Preprocess Finished."
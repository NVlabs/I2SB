
NAME=$1

echo "Note: available models are inpaint-center, inpaint-freeform1020, inpaint-freeform2030, blur-gauss, blur-uni, sr4x-pool, sr4x-bicubic, jpeg-5, and jpeg-10"
echo "Specified [$NAME]"

mkdir -p results && cd results

if [ "$NAME" == inpaint-center ]; then
    gdown --folder 1ounBKyUezkZmjC7d3tzbBwfEBe0Y7bkM
fi
if [ "$NAME" == inpaint-freeform1020 ]; then
    gdown --folder 1WGYzrxO-t4Q6XadgwWdUuh6Ue966LYKw
fi
if [ "$NAME" == inpaint-freeform2030 ]; then
    gdown --folder 18Q1ZtUcl3z63v0VOSR5hVH3T5fGNgf3I
fi
if [ "$NAME" == blur-gauss ]; then
    gdown --folder 1vatmaOiEgO_Z9JQiFhHdcjiSNgnS9lBY
fi
if [ "$NAME" == blur-uni ]; then
    gdown --folder 1_U9IhgH7CxNp0_8o9F3D86sihflPGKDZ
fi
if [ "$NAME" == sr4x-pool ]; then
    gdown --folder 1VIA8haw3Oy0OqL-CeEM2Rekk92tL0gD0
fi
if [ "$NAME" == sr4x-bicubic ]; then
    gdown --folder 1QfpggccSV9llWMBrklZ6OPdlGR7qRd6F
fi
if [ "$NAME" == jpeg-5 ]; then
    gdown --folder 17bPMUX5yBcL8YTN8nbfJT6WHXU6CNLd7
fi
if [ "$NAME" == jpeg-10 ]; then
    gdown --folder 13on7v54W8ncQ4RlSl3PIZn0NalfiGaJd
fi

#!python3
import os
import urllib.request
import urllib.error

import svgutils.transform as sg
import re
import argparse
import segno


def generate_qr(url: str, qr_image_path: str):
    qrcode = segno.make_qr(url)
    qrcode.save(qr_image_path, dark="515151")


def qr_code_size(qr_image_path):
    with open(qr_image_path, "r") as qr_img:
        content = qr_img.read()

    width = re.findall(r'width="(\d+)"', content)[0]
    height = re.findall(r'height="(\d+)"', content)[0]
    sizes = [width, height]
    assert len(sizes) == 2 and sizes[0] == sizes[1]

    with open(qr_image_path, "w+") as qr_img:
        qr_img.write(content)

    return float(width)


def combine_svg(background_path, qr_size, qr_image_path, output_path):
    background = sg.fromfile(background_path)
    width_background, height_background = map(float, background.get_size())

    scale = 0.72 * width_background / qr_size
    size_qr = qr_size * scale

    qr_code = sg.fromfile(qr_image_path)

    plot1 = qr_code.getroot()
    # get the plot objects
    plot1.moveto(0, 0, scale_x=scale, scale_y=scale)

    plot1.moveto(width_background / 2 - size_qr / 2, height_background / 2 - size_qr / 2)

    # append plots and labels to figure
    background.append([plot1])

    print(f"Saving file to {os.getcwd()}/{output_path}")
    # save generated SVG files
    background.save(output_path)


def check_and_download_files():
    files = ['sticker_dark.svg', 'sticker_light.svg']
    url = "https://raw.githubusercontent.com/norlab-ulaval/visualIdentity/v1.0.0/qr_code/templates/"
    for file in files:
        if not os.path.isfile('/tmp/' + file):
            try:
                urllib.request.urlretrieve(url + file, '/tmp/' + file)
            except urllib.error.HTTPError as e:
                raise Exception(f"Couldn't download background .svg files: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://norlab.ulaval.ca",
                        type=str, help="URL to generate the QR code from")
    args = parser.parse_args()
    url = args.url
    qr_image_path = "/tmp/out.svg"
    check_and_download_files()
    generate_qr(url=url, qr_image_path=qr_image_path)

    qr_size = qr_code_size(qr_image_path)
    combine_svg('/tmp/sticker_dark.svg', qr_size, qr_image_path, os.path.join(os.getcwd(), "qr_dark.svg"))
    combine_svg('/tmp/sticker_light.svg', qr_size, qr_image_path, os.path.join(os.getcwd(), "qr_light.svg"))


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import requests
import shutil
from urllib.parse import urlparse

from utils import coerce_to_path_and_create_dir
from utils.logger import print_info, print_error


class IIIFDownloader:
    """Download all image resources from a list of manifest urls."""

    def __init__(self, manifest_urls, output_dir, width=None, height=None):
        self.manifest_urls = manifest_urls
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.size = self.get_formatted_size(width, height)

    @staticmethod
    def get_formatted_size(width, height):
        if width is None:
            size = 'full' if height is None else ',{}'.format(height)
        else:
            size = '{},'.format(width) if height is None else '{},{}'.format(width, height)
        return size

    def run(self):
        for url in self.manifest_urls:
            manifest = self.get_json(url)
            if manifest is not None:
                manifest_id = Path(urlparse(manifest['@id']).path).parent.name
                print_info('Processing {}...'.format(manifest_id))
                output_path = coerce_to_path_and_create_dir(self.output_dir / manifest_id)
                resources = self.get_resources(manifest)

                for resource_url in resources:
                    resource_url = '/'.join(resource_url.split('/')[:-3] + [self.size] + resource_url.split('/')[-2:])
                    with requests.get(resource_url, stream=True) as response:
                        response.raw.decode_content = True
                        resrc_path = Path(urlparse(resource_url).path)
                        name = '{}{}'.format(resrc_path.parts[-5], resrc_path.suffix)
                        output_file = output_path / name
                        print_info('Saving {}...'.format(output_file.relative_to(self.output_dir)))
                        with open(output_file, mode='wb') as f:
                            shutil.copyfileobj(response.raw, f)

    @staticmethod
    def get_json(url):
        try:
            response = requests.get(url)
            if response.ok:
                return response.json()
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print_error(e)
            return None

    @staticmethod
    def get_resources(manifest):
        try:
            canvases = manifest['sequences'][0]['canvases']
            images_list = [canvas['images'] for canvas in canvases]
            return [image['resource']['@id'] for images in images_list for image in images]
        except KeyError as e:
            print_error(e)
            return []

    @staticmethod
    def get_resources_with_targets(manifest):
        try:
            canvases = manifest['sequences'][0]['canvases']
            images_list = [canvas['images'] for canvas in canvases]
            return [(image['resource']['@id'], image['on']) for images in images_list for image in images]
        except KeyError as e:
            print_error(e)
            return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download all image resources from a list of manifest urls')
    parser.add_argument('-f', '--file', nargs='?', type=str, required=True, help='File containing manifest urls')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, default='output', help='Output directory name')
    parser.add_argument('--width', type=int, default=None, help='Image width')
    parser.add_argument('--height', type=int, default=None, help='Image height')
    args = parser.parse_args()

    with open(args.file, mode='r') as f:
        manifest_urls = f.read().splitlines()
    manifest_urls = list(filter(None, manifest_urls))

    output_dir = args.output_dir if args.output_dir is not None else 'output'
    downloader = IIIFDownloader(manifest_urls, output_dir=output_dir, width=args.width, height=args.height)
    downloader.run()

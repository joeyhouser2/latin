"""
IIIF Manuscript Downloader

Download manuscript images from Vatican, Bodleian, and other IIIF-compliant repositories.

Usage:
    python iiif_downloader.py --manifest URL --output DIR
    python iiif_downloader.py --vatican "Vat.lat.3773" --output DIR
    python iiif_downloader.py --bodleian "MS. Laud Misc. 509" --output DIR
"""

import argparse
import json
import requests
import time
from pathlib import Path
from urllib.parse import quote
from typing import Optional, List
import re


class IIIFDownloader:
    """Download images from IIIF manifests."""
    
    def __init__(self, delay: float = 0.5, max_retries: int = 3):
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LatinRAG-Research/1.0 (scholarly research)"
        })
    
    def get_manifest(self, url: str) -> dict:
        """Fetch and parse IIIF manifest."""
        print(f"Fetching manifest: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def extract_image_urls(self, manifest: dict) -> List[dict]:
        """Extract image URLs from manifest (handles IIIF 2.x and 3.x)."""
        images = []
        
        # IIIF 2.x structure
        sequences = manifest.get("sequences", [])
        if sequences:
            canvases = sequences[0].get("canvases", [])
            for i, canvas in enumerate(canvases):
                canvas_images = canvas.get("images", [])
                if canvas_images:
                    resource = canvas_images[0].get("resource", {})
                    image_info = self._extract_image_info_v2(resource, i)
                    if image_info:
                        images.append(image_info)
        
        # IIIF 3.x structure
        items = manifest.get("items", [])
        for i, item in enumerate(items):
            body = item.get("items", [{}])[0].get("items", [{}])[0].get("body", {})
            image_info = self._extract_image_info_v3(body, i)
            if image_info:
                images.append(image_info)
        
        return images
    
    def _extract_image_info_v2(self, resource: dict, index: int) -> Optional[dict]:
        """Extract image info from IIIF 2.x resource."""
        if "service" in resource:
            service = resource["service"]
            service_id = service.get("@id", service.get("id", ""))
            if service_id:
                return {
                    "index": index,
                    "service_url": service_id,
                    "format": "jpg"
                }
        
        # Direct image URL
        image_id = resource.get("@id", resource.get("id", ""))
        if image_id:
            return {
                "index": index,
                "direct_url": image_id,
                "format": "jpg"
            }
        
        return None
    
    def _extract_image_info_v3(self, body: dict, index: int) -> Optional[dict]:
        """Extract image info from IIIF 3.x body."""
        if "service" in body:
            services = body["service"]
            if isinstance(services, list) and services:
                service = services[0]
            else:
                service = services
            
            service_id = service.get("id", service.get("@id", ""))
            if service_id:
                return {
                    "index": index,
                    "service_url": service_id,
                    "format": "jpg"
                }
        
        image_id = body.get("id", "")
        if image_id:
            return {
                "index": index,
                "direct_url": image_id,
                "format": "jpg"
            }
        
        return None
    
    def build_image_url(self, image_info: dict, size: str = "1000,") -> str:
        """
        Build full IIIF image URL.
        
        Args:
            image_info: Dict with service_url or direct_url
            size: IIIF size parameter (e.g., "full", "1000,", "pct:50")
        """
        if "direct_url" in image_info:
            url = image_info["direct_url"]
            # If it's already a full image URL, return as-is
            if url.endswith((".jpg", ".png", ".tif")):
                return url
            # Otherwise, construct IIIF URL
            return f"{url}/full/{size}/0/default.jpg"
        
        if "service_url" in image_info:
            service = image_info["service_url"]
            return f"{service}/full/{size}/0/default.jpg"
        
        raise ValueError("No valid URL in image_info")
    
    def download_image(self, url: str, output_path: Path) -> bool:
        """Download a single image with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    output_path.write_bytes(response.content)
                    return True
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = int(response.headers.get("Retry-After", 60))
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  HTTP {response.status_code}")
                    return False
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(self.delay * (attempt + 1))
        
        return False
    
    def download_manifest(
        self,
        manifest_url: str,
        output_dir: str,
        max_images: Optional[int] = None,
        size: str = "1000,",
        start_page: int = 0
    ) -> List[str]:
        """
        Download all images from an IIIF manifest.
        
        Args:
            manifest_url: URL to IIIF manifest
            output_dir: Directory to save images
            max_images: Optional limit
            size: IIIF size parameter
            start_page: Skip this many pages
        
        Returns:
            List of downloaded file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Fetch manifest
        manifest = self.get_manifest(manifest_url)
        
        # Save manifest for reference
        manifest_file = output_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        
        # Extract image URLs
        images = self.extract_image_urls(manifest)
        print(f"Found {len(images)} pages")
        
        # Apply limits
        images = images[start_page:]
        if max_images:
            images = images[:max_images]
        
        # Download
        downloaded = []
        for i, img_info in enumerate(images):
            page_num = img_info["index"]
            filename = output_path / f"page_{page_num:04d}.jpg"
            
            url = self.build_image_url(img_info, size)
            print(f"Downloading page {page_num} ({i+1}/{len(images)})...")
            
            if self.download_image(url, filename):
                downloaded.append(str(filename))
            
            time.sleep(self.delay)
        
        print(f"\nDownloaded {len(downloaded)} images to {output_dir}")
        return downloaded


# ============================================================================
# REPOSITORY-SPECIFIC HELPERS
# ============================================================================

def get_vatican_manifest(shelfmark: str) -> str:
    """
    Get Vatican IIIF manifest URL from shelfmark.
    
    Examples:
        "Vat.lat.3773" -> Virgil manuscript
        "Pal.lat.1631" -> Palatine collection
        "Barb.lat.4" -> Barberini collection
    """
    # Normalize shelfmark
    shelfmark = shelfmark.strip()
    # Vatican uses MSS_ prefix
    formatted = f"MSS_{shelfmark.replace(' ', '.')}"
    return f"https://digi.vatlib.it/iiif/{formatted}/manifest.json"


def get_bodleian_manifest(shelfmark: str) -> str:
    """
    Get Bodleian IIIF manifest URL from shelfmark.
    
    Examples:
        "MS. Laud Misc. 509"
        "MS. Bodl. 264"
    """
    # Bodleian uses Digital.Bodleian
    # This is a simplified lookup - actual URLs vary
    encoded = quote(shelfmark.replace(" ", "_").replace(".", "_"))
    return f"https://iiif.bodleian.ox.ac.uk/iiif/manifest/{encoded}.json"


def get_bnf_manifest(ark: str) -> str:
    """
    Get BnF Gallica IIIF manifest URL from ARK identifier.
    
    Example:
        "ark:/12148/btv1b8432895r"
    """
    ark_id = ark.replace("ark:/", "").replace("/", "_")
    return f"https://gallica.bnf.fr/iiif/{ark}/manifest.json"


def get_ecodices_manifest(collection: str, manuscript: str) -> str:
    """
    Get e-codices IIIF manifest URL.
    
    Example:
        collection="csg", manuscript="0390"  -> St. Gallen, Cod. Sang. 390
    """
    return f"https://www.e-codices.unifr.ch/metadata/iiif/{collection}-{manuscript}/manifest.json"


def search_vatican_manuscripts(query: str) -> List[dict]:
    """
    Search Vatican Library catalog (basic implementation).
    
    Note: This is a simplified search. The actual Vatican catalog
    requires more complex queries.
    """
    # This would need to scrape/query the Vatican OPAC
    # For now, return some known Latin manuscripts
    known_manuscripts = [
        {"shelfmark": "Vat.lat.3225", "title": "Vergilius Vaticanus (5th c.)"},
        {"shelfmark": "Vat.lat.3867", "title": "Vergilius Romanus (5th c.)"},
        {"shelfmark": "Vat.lat.3773", "title": "Opera Vergilii"},
        {"shelfmark": "Pal.lat.1631", "title": "Lorsch Gospels"},
        {"shelfmark": "Reg.lat.316", "title": "Gelasian Sacramentary"},
    ]
    
    query_lower = query.lower()
    return [m for m in known_manuscripts if query_lower in m["title"].lower() or query_lower in m["shelfmark"].lower()]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download manuscript images from IIIF repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download from manifest URL
    python iiif_downloader.py --manifest https://digi.vatlib.it/iiif/MSS_Vat.lat.3773/manifest.json --output vat_lat_3773
    
    # Download Vatican manuscript by shelfmark
    python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773
    
    # Download first 10 pages only
    python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773 --max 10
    
    # Download at full resolution (slower, larger files)
    python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773 --size full
        """
    )
    
    # Source options (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Direct IIIF manifest URL")
    source.add_argument("--vatican", help="Vatican Library shelfmark (e.g., Vat.lat.3773)")
    source.add_argument("--bodleian", help="Bodleian Library shelfmark")
    source.add_argument("--bnf", help="BnF Gallica ARK identifier")
    source.add_argument("--ecodices", help="e-codices identifier (collection-manuscript)")
    source.add_argument("--search-vatican", help="Search Vatican catalog")
    
    # Output options
    parser.add_argument("--output", "-o", required=False, help="Output directory")
    parser.add_argument("--max", type=int, help="Maximum number of pages to download")
    parser.add_argument("--start", type=int, default=0, help="Start from this page number")
    parser.add_argument("--size", default="1000,", help="IIIF size parameter (default: 1000,)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    
    args = parser.parse_args()
    
    # Handle search
    if args.search_vatican:
        results = search_vatican_manuscripts(args.search_vatican)
        if results:
            print(f"Found {len(results)} manuscripts:")
            for r in results:
                print(f"  {r['shelfmark']}: {r['title']}")
        else:
            print("No results found")
        return
    
    # Determine manifest URL
    if args.manifest:
        manifest_url = args.manifest
        default_output = "manuscript_download"
    elif args.vatican:
        manifest_url = get_vatican_manifest(args.vatican)
        default_output = args.vatican.replace(".", "_").replace(" ", "_")
    elif args.bodleian:
        manifest_url = get_bodleian_manifest(args.bodleian)
        default_output = args.bodleian.replace(".", "_").replace(" ", "_")
    elif args.bnf:
        manifest_url = get_bnf_manifest(args.bnf)
        default_output = args.bnf.replace("/", "_").replace(":", "_")
    elif args.ecodices:
        parts = args.ecodices.split("-", 1)
        if len(parts) != 2:
            parser.error("e-codices format should be 'collection-manuscript' (e.g., csg-0390)")
        manifest_url = get_ecodices_manifest(parts[0], parts[1])
        default_output = args.ecodices
    
    output_dir = args.output or default_output
    
    # Download
    downloader = IIIFDownloader(delay=args.delay)
    
    try:
        downloaded = downloader.download_manifest(
            manifest_url=manifest_url,
            output_dir=output_dir,
            max_images=args.max,
            size=args.size,
            start_page=args.start
        )
        
        print(f"\n✓ Downloaded {len(downloaded)} images")
        print(f"  Output: {output_dir}/")
        print(f"  Manifest saved: {output_dir}/manifest.json")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        print("  The manuscript may not be digitized or the URL format may have changed.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
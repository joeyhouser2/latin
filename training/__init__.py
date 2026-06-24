"""Tools for building parallel corpora and (later) fine-tuning era-specific
Latin/Greek -> English translation models.

The bottleneck for training a translator is aligned source/target pairs. This
package mines them from sources we already reach: Perseus dual editions (a `-lat`
or `-grc` text and its `-eng` translation share a CTS citation scheme), and other
aligned corpora. Output is era-tagged JSONL ready for fine-tuning.
"""

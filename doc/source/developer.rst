.. _developer:

===================================
The developer and maintainer corner
===================================

This chapter is a kind ow unstructured "howto" that states the various development and maintenance
rules for the :term:`cots` project. The poor man's quality assurance plan.

This chapter is opened to be restructured in the future.

Git workflow and life cycle
===========================

The "master" git branch
-----------------------

The "master" branch contains the source trees from which the public releases of the :term:`cots`
software products are built and distributed to the registries (PyPI, Docker hub, doc site).

Each release is identified by a Git tag named with the software distribution version.

The changes in the "master" branch are processed by merging the "testing" branch when all issues
from a milestone are considered as fixed, tested and relevant documentation changes are written.

.. warning::

   The only change allowed directly in this branch is the :file:`VERSION.txt` file.

The "testing" branch
--------------------

The "testing" branch contains the "bleeding edge" version of the :term:`cots` software, tests and
documentation.

The changes in the "testing" branch are processed by merging a personal branch when the issues from the personal branch are considered closed, and **the unit tests don't report any failure**.

.. warning::

   Only "quick fix" changes are allowed directly in the "testing" branch.

Personal branches
-----------------

Personal branches are created by forking the **testing** branch, by individual developers in order
to fix one or more issues.

The release cycle
-----------------

.. rubric:: Step 1: fork the "testing" branch in a personal branch

Refresh your clone of the testing branch:

.. code:: console

   git checkout testing
   git pull

.. rubric:: Step 2: create your personal branch:

.. code:: console

   git checkout -b <your-new-personal-branch>

.. hint::

   The name of the branch must contain your name or nick, a
   very short title of the planned change and the issue numbers (usually only one).

   Example: `glenfant-cmdline-24`

.. rubric:: Step 3: Share your changes to the "testing" branch

Your changes are done and tested. Time to share your work. You should first squash your commits in
order to keep only the relevant ones. This `blog article
<https://www.ekino.com/articles/comment-squasher-efficacement-ses-commits-avec-git>`__ explains why
and how to do it.

The "testing" branch may have changed since you forked it at step 2. In that case, you must replay
these changes into your personal branch.

.. code:: console

   git co testing
   git pull
   git co <your-personal-branch>
   git rebase testing

.. important::

   This "rebase" operation may introduce silent conflicts. So make sure the unit tests don't raise
   any failure after rebasing.

.. code:: console

   git push

Point your browser to the |vcs_server|, then in the :menuselection:`CI / CD --> Pipelines` menu
selection. You should see the progress of the CI / CD operations. Once the CI steps are successfully
executed, you should issue a new **merge request** to share your work with your teammates.

.. hint::

   The benefits of a **merge request** over a direct Git merge operation are:

   - All merges to "testing" are recorded in a shared place.
   - You may ask for a code review from your teammates **before** the merge is executed.
   - A "dry run" merge is executed such it identifies potential conflicts.
   - You and your teammates may browse the changes in details using the links in the merge request
     page.

Once the merge request is executed, you must inspect the CI / CD pipeline, and fix potential issues
directly in the "testing" branch.

Coding style
============

Code style
----------

Code style must comply with |pep_8| requirements.

Python IDEs like PyCharm or Microsoft VSCode have an "autopep8" feature that helps you to keep this
rule respected.

Comments
--------

Inline comments (`# ...`) must say **what the code does** in a high level view and not explain the
language.

Bad examples:

.. code:: python

   # Increment the counter
   counter += 1

   # "+=" is the "add right to left" operator
   counter += 1

Good example:

.. code:: python

   # Prepare to visit next item
   counter += 1

The "good ratio" is about 1 line comment for 10 lines of code.

Type hints (PEP 484)
--------------------

All functions and methods that take part of the public API must use type hints as described by the
|pep_484| document.

This is an essential help for the code documentation as well as an IDE helper for completions.

Example:

.. code:: python

   def is_closed(handle: int) -> bool:
       """
       Checks a stuff is closed
       ...
       """

Docstrings
----------

Docstrings of **public API** resources will be written in ReStructuredText and may be processed by
Sphinx exactly like the lines you are reading.

In addition, the descriptions of relevant arguments, keyword arguments, attributes, exceptions,
(...) will be expressed using the `Google style
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

Example:

.. code:: python

   from decimal import Decimal

   class BankAccount:
       """Person or company bank account

       Attributes:
           holder: The possessing person or company.
           balance: actual account amount
       """
       def __init__(self, holder: AccountHolder) -> None:
           self.holder = holder
           self.balance: Decimal = Decimal("0.00")

       def credit(self, value: Decimal) -> Decimal:
           """Add a value to the balance

           Args:
               value: to be added to the balance (may be < 0)

           Returns:
               Updated balance

           Raises:
               ValueError: if the resulting amount below 0
           """
           if self.balance + value < 0:
               raise ValueError("Negative balance is forbidden, operation will be canceled")
           self.balance += value
           return self.balance

Running the tests
=================

The unit tests are executed through the :command:`pytest` command. The default options sit in the
:file:`setup.cfg` file. The tests codes and resources are hosted in the :file:`tests/...` directory.

As above stated, the unit tests are powered by the third party `pytest
<https://docs.pytest.org/en/latest/>`_ tool.

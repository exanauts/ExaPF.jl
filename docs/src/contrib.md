# How to contribute to ExaPF

Welcome, and thank you for your interest in contributing to ExaPF. This document provides a roadmap of some of the ways you can help improve this project.

## Code of Conduct

All participants in this project are expected to uphold the spirit of open collaboration and respect. By participating, you agree to maintain a positive and inclusive environment.

## Improve the Documentation

If you've asked a question or provided an answer on our community forum, chances are there's an opportunity to improve our documentation. Your firsthand experience makes you uniquely qualified to clarify and enhance the information we provide to users.

The ExaPF documentation is written in Markdown and built using the native Julia documentation system. If you spot a typo or a small error, you can easily fix it through GitHub's online editor. For larger changes, you'll want to edit locally and submit a pull request.

## Report Bugs

One of the most valuable contributions you can make is identifying and reporting bugs. Before posting a bug report, we recommend discussing the issue on our community forum. This can help clarify whether it's a real bug or perhaps a misunderstanding.

## Contribute Code to ExaPF

Contributing code to ExaPF is a fantastic way to enhance the project and learn more about Julia and high-performance computing. 

If you're new to Git, GitHub, or Julia development, there are plenty of resources available online to get you started. Don't hesitate to ask for help if you need it.

To contribute code to ExaPF, follow these steps:

**Step 1: Identify a Task**

Start by finding an open issue that resonates with you. Discuss your proposed solution with other contributors to ensure your efforts align with the project's direction.

**Step 2: Fork ExaPF**

Visit the ExaPF repository and click on the "Fork" button to create your personal copy of the project.

**Step 3: Install ExaPF Locally**

Install your fork of ExaPF on your local machine using Julia's package manager.

**Step 4: Create a New Branch**

Create a new branch in your fork for the changes you're planning to make.

**Step 5: Make Changes**

Now you're ready to make changes to the ExaPF source code. Make sure to follow our style guide and add tests and documentation for any modifications or new features you implement.

**Step 6: Test Your Changes**

Test your changes by running the ExaPF test suite. This will help ensure your modifications don't inadvertently break any existing functionality.

**Step 7: Submit a Pull Request**

After testing your changes, push your branch to your GitHub fork and create a pull request. Remember, it's important to provide a clear and concise explanation of your changes when submitting a pull request.

**Step 8: Respond to Feedback**

Once your pull request is submitted, it will be reviewed by the ExaPF maintainers. Be open to feedback and ready to make revisions if necessary. A constructive dialogue will help ensure the quality and consistency of the ExaPF code# How to Contribute to ExaPF

Welcome! We're delighted to hear that you're interested in contributing to ExaPF. This guide will walk you through the different ways in which you can contribute to this project.

## Code of Conduct

We want to foster an inclusive and respectful environment in this project. Everyone participating in ExaPF is expected to abide by our [Code of Conduct](https://github.com/exanauts/ExaPF.jl/blob/master/CODE_OF_CONDUCT.md). By engaging with our project, you agree to uphold these guidelines.

## Enhance Documentation

Insights from your experience using ExaPF can be very useful in improving the [documentation](https://exapf.dev/ExaPF.jl/dev/). If you found something unclear or missing while using ExaPF, you're probably the best person to enhance the relevant documentation.

Our documentation is written in Markdown and we use [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) for building it. You can find the source files for the documentation [here](https://github.com/exanauts/ExaPF.jl/tree/master/docs).

For minor changes such as fixing typos or making small clarifications, you can use GitHub's online editor. If your changes are substantial or affect multiple files, you'll need to clone the repository locally and submit a pull request.

## Report Bugs

If you encounter a problem with ExaPF, you can contribute by filing a [bug report](https://github.com/exanauts/ExaPF.jl/issues/new?template=bug_report.md). Make sure to read the provided guidelines before submitting your report.

## Contribute Code to ExaPF

We're always open to code contributions. However, if you're new to Git, GitHub, and Julia development, you might find the initial steps challenging. Fortunately, there are numerous online tutorials available, including guides for [GitHub](https://guides.github.com/activities/hello-world/), [Git and GitHub](https://try.github.io/), [Git](https://git-scm.com/book/en/v2), and [Julia package development](https://docs.julialang.org/en/v1/stdlib/Pkg/#Developing-packages-1).

The workflow for contributing code to ExaPF is outlined below:

1. **Identify What to Work On**  
   Identify an [open issue](https://github.com/exanauts/ExaPF.jl/issues) or create a new one.

2. **Fork the Repository**
    - Go to the GitHub page for the repository you want to contribute to.
    - Click the "Fork" button at the top right of the page.
    - This will create a copy of the repository in your own GitHub account.

3. **Clone the Repository to Your Local Machine**
    - Navigate to your fork of the repository on your GitHub account.
    - Click the "Code" button and then click the clipboard icon to copy the repository URL.
    - Open a terminal/command prompt on your local machine.
    - Navigate to the directory where you want to store the project.
    - Run `git clone <URL>` where `<URL>` is the URL you copied earlier.
    - Now you have a local copy of the project on your machine.

4. **Create a New Branch**
    - In your terminal, navigate to the root directory of the project.
    - Run `git checkout -b <branch-name>` to create a new branch and switch to it. `<branch-name>` should be a descriptive name for the changes you plan to make.

5. **Make Your Changes**
    - Now you can start making changes to the project. Make sure your changes are in line with the project's contribution guidelines.
    - Make sure to write clean, understandable code and include comments where necessary.

6. **Commit Your Changes**
    - Once you have made all your changes, it's time to commit them.
    - Run `git add .` to stage all your changes.
    - Run `git commit -m "<commit-message>"` to commit your changes. `<commit-message>` should be a short, descriptive message of what changes were made.

7. **Push Your Changes to GitHub**
    - Now that your changes are committed, you need to push them to GitHub.
    - Run `git push origin <branch-name>` to push your changes to your forked repository.

8. **Open a Pull Request**
    - Navigate to the GitHub page for your forked repository.
    - Click "New pull request".
    - Make sure the base repository is the repository you forked from and the compare repository is your fork.
    - Click "Create pull request".
    - Fill in the title and description for your pull request and click "Create pull request".
    - Now your changes have been submitted for review!

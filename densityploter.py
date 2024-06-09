import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.stats import norm, expon, poisson, binom, uniform, beta, gamma, chi2

class DistributionPlotter:
    def __init__(self, figsize=(15, 15)):
        self.figsize = figsize
        self._create_widgets()
        self._create_interactive_output()

    def plot_normal_distribution(self, mean=0, std_dev=1, color='red', ax=None):
        x = np.linspace(-10, 10, 1000)
        mean = round(mean, 4)
        std_dev = round(std_dev, 4)
        y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        ax.plot(x, y, color=color, label=f'Mean: {mean}, Std Dev: {std_dev}')
        return x, y

    def plot_exponential_distribution(self, lmbda=1, color='red', ax=None):
        x = np.linspace(0, 10, 1000)
        y = lmbda * np.exp(-lmbda * x)
        ax.plot(x, y, color=color, label=f'Lambda: {lmbda}')
        return x, y

    def plot_poisson_distribution(self, lmbda=1, color='red', ax=None):
        x = np.arange(0, 20)
        y = poisson.pmf(x, lmbda)
        ax.stem(x, y, basefmt=" ", linefmt=color, markerfmt= "o", label=f'Lambda: {lmbda}' )
        return x, y

    def plot_binomial_distribution(self, n=10, p=0.5, color='red', ax=None):
        x = np.arange(0, n + 1)
        y = binom.pmf(x, n, p)
        ax.stem(x, y, basefmt=" ", linefmt=color, markerfmt= "o", label=f'n: {n}, p: {p}' )
        return x, y

    def plot_uniform_distribution(self, a=1, b=0, color='red', ax=None):
        x = np.linspace(0, 2, 1000)
        a = round(a,2)
        b = round(b,2)
        y = uniform.pdf(x, loc=a, scale=b - a)
        ax.plot(x, y, color=color, label=f'a: {a}, b: {b}')
        return x, y

    def plot_beta_distribution(self, a=2, b=2, color='red', ax=None):
        x = np.linspace(0, 1, 1000)
        a = round(a,5)
        b = round(b,5)
        y = beta.pdf(x, a, b)
        ax.plot(x, y, color=color, label=f'a: {a}, b: {b}')
        return x, y

    def plot_gamma_distribution(self, a=2, color='red', ax=None):
        x = np.linspace(0, 20, 1000)
        y = gamma.pdf(x, a)
        ax.plot(x, y, color=color, label=f'a: {a}')
        return x, y

    def plot_chi2_distribution(self, df=2, color='red', ax=None):
        x = np.linspace(0, 20, 1000)
        y = chi2.pdf(x, df)
        ax.plot(x, y, color=color, label=f'Degrees of Freedom: {df}')
        return x, y

    def plot_distribution(self, distribution, color='red', mean=0, std_dev=1, lmbda=1, n=10, p=0.5, a=2, b=2, df=2, ax=None):
        if distribution == 'Normal':
            return self.plot_normal_distribution(mean, std_dev, color, ax)
        elif distribution == 'Exponential':
            return self.plot_exponential_distribution(lmbda, color, ax)
        elif distribution == 'Poisson':
            return self.plot_poisson_distribution(lmbda, color, ax)
        elif distribution == 'Binomial':
            return self.plot_binomial_distribution(n, p, color, ax)
        elif distribution == 'Uniform':
            return self.plot_uniform_distribution(a, b, color, ax)
        elif distribution == 'Beta':
            return self.plot_beta_distribution(a, b, color, ax)
        elif distribution == 'Gamma':
            return self.plot_gamma_distribution(a, color, ax)
        elif distribution == 'Chi-Square':
            return self.plot_chi2_distribution(df, color, ax)

    def plot_combined_distribution(self, dist1, dist2, ax=None):
        ax.plot(dist1[0], dist1[1], color='red', alpha=0.5, label='Distribution 1')
        ax.plot(dist2[0], dist2[1], color='blue', alpha=0.5, label='Distribution 2')
        ax.set_title('Combined Distribution (Overlay)')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density / Mass')
        ax.legend(loc='upper left')
        ax.grid(True)

    def plot_final_distribution(self, dist1, dist2, ax=None):
        x = np.union1d(dist1[0], dist2[0])
        y = np.interp(x, dist1[0], dist1[1]) + np.interp(x, dist2[0], dist2[1])
        ax.plot(x, y, color='purple', label='Final Combined Distribution')
        ax.set_title('Final Combined Distribution')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density / Mass')
        ax.legend(loc='upper right')
        ax.grid(True)

    def update_plots(self, distribution1, mean1, std_dev1, lmbda1, n1, p1, a1, b1, df1,
                     distribution2, mean2, std_dev2, lmbda2, n2, p2, a2, b2, df2):
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)

        # Plot Distribution 1
        dist1 = self.plot_distribution(distribution1, 'red', mean1, std_dev1, lmbda1, n1, p1, a1, b1, df1, ax=axs[0, 0])
        axs[0, 0].set_title(f'{distribution1} Distribution')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('Probability Density / Mass')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True)

        # Plot Distribution 2
        dist2 = self.plot_distribution(distribution2, 'blue', mean2, std_dev2, lmbda2, n2, p2, a2, b2, df2, ax=axs[0, 1])
        axs[0, 1].set_title(f'{distribution2} Distribution')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('Probability Density / Mass')
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].grid(True)

        # Plot Combined Distribution (Overlay)
        self.plot_combined_distribution(dist1, dist2, ax=axs[1, 0])

        # Plot Final Combined Distribution
        self.plot_final_distribution(dist1, dist2, ax=axs[1, 1])
        plt.show()

    def _create_widgets(self):
        self.distribution1_dropdown = widgets.Dropdown(
            options=['Normal', 'Exponential', 'Poisson', 'Binomial', 'Uniform', 'Beta', 'Gamma', 'Chi-Square'],
            value='Normal',
            description='Distribution 1:'
        )
        self.distribution2_dropdown = widgets.Dropdown(
            options=['Normal', 'Exponential', 'Poisson', 'Binomial', 'Uniform', 'Beta', 'Gamma', 'Chi-Square'],
            value='Beta',
            description='Distribution 2:'
        )
        self.mean1_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean 1:')
        self.std_dev1_slider = widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Std Dev 1:')
        self.lmbda1_slider = widgets.FloatSlider(value=1, min=0.1, max=10, step=0.1, description='Lambda 1:')
        self.n1_slider = widgets.IntSlider(value=10, min=1, max=100, step=1, description='n 1:')
        self.p1_slider = widgets.FloatSlider(value=0.5, min=0.01, max=1, step=0.01, description='p 1:')
        self.a1_slider = widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='a 1:')
        self.b1_slider = widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='b 1:')
        self.df1_slider = widgets.IntSlider(value=2, min=1, max=20, step=1, description='df 1:')

        self.mean2_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean 2:')
        self.std_dev2_slider = widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Std Dev 2:')
        self.lmbda2_slider = widgets.FloatSlider(value=1, min=0.1, max=10, step=0.1, description='Lambda 2:')
        self.n2_slider = widgets.IntSlider(value=10, min=1, max=100, step=1, description='n 2:')
        self.p2_slider = widgets.FloatSlider(value=0.5, min=0.01, max=1, step=0.01, description='p 2:')
        self.a2_slider = widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='a 2:')
        self.b2_slider = widgets.FloatSlider(value=3, min=0.1, max=10, step=0.1, description='b 2:')
        self.df2_slider = widgets.IntSlider(value=2, min=1, max=20, step=1, description='df 2:')

        self.ui1 = widgets.VBox([
            self.distribution1_dropdown,
            self.mean1_slider,
            self.std_dev1_slider,
            self.lmbda1_slider,
            self.n1_slider,
            self.p1_slider,
            self.a1_slider,
            self.b1_slider,
            self.df1_slider
        ])

        self.ui2 = widgets.VBox([
            self.distribution2_dropdown,
            self.mean2_slider,
            self.std_dev2_slider,
            self.lmbda2_slider,
            self.n2_slider,
            self.p2_slider,
            self.a2_slider,
            self.b2_slider,
            self.df2_slider
        ])

        self.ui = widgets.HBox([self.ui1, self.ui2])

    def _create_interactive_output(self):
        self.out = widgets.interactive_output(self.update_plots, {
            'distribution1': self.distribution1_dropdown,
            'mean1': self.mean1_slider,
            'std_dev1': self.std_dev1_slider,
            'lmbda1': self.lmbda1_slider,
            'n1': self.n1_slider,
            'p1': self.p1_slider,
            'a1': self.a1_slider,
            'b1': self.b1_slider,
            'df1': self.df1_slider,
            'distribution2': self.distribution2_dropdown,
            'mean2': self.mean2_slider,
            'std_dev2': self.std_dev2_slider,
            'lmbda2': self.lmbda2_slider,
            'n2': self.n2_slider,
            'p2': self.p2_slider,
            'a2': self.a2_slider,
            'b2': self.b2_slider,
            'df2': self.df2_slider
        })

    def update_ui(self, *args):
        distribution1 = self.distribution1_dropdown.value
        distribution2 = self.distribution2_dropdown.value
        self.mean1_slider.layout.display = 'none' if distribution1 != 'Normal' else 'inline-flex'
        self.std_dev1_slider.layout.display = 'none' if distribution1 != 'Normal' else 'inline-flex'
        self.lmbda1_slider.layout.display = 'none' if distribution1 not in ['Exponential', 'Poisson'] else 'inline-flex'
        self.n1_slider.layout.display = 'none' if distribution1 != 'Binomial' else 'inline-flex'
        self.p1_slider.layout.display = 'none' if distribution1 != 'Binomial' else 'inline-flex'
        self.a1_slider.layout.display = 'none' if distribution1 not in ['Beta', 'Gamma', 'Uniform'] else 'inline-flex'
        self.b1_slider.layout.display = 'none' if distribution1 not in ['Beta', 'Uniform'] else 'inline-flex'
        self.df1_slider.layout.display = 'none' if distribution1 != 'Chi-Square' else 'inline-flex'

        self.mean2_slider.layout.display = 'none' if distribution2 != 'Normal' else 'inline-flex'
        self.std_dev2_slider.layout.display = 'none' if distribution2 != 'Normal' else 'inline-flex'
        self.lmbda2_slider.layout.display = 'none' if distribution2 not in ['Exponential', 'Poisson'] else 'inline-flex'
        self.n2_slider.layout.display = 'none' if distribution2 != 'Binomial' else 'inline-flex'
        self.p2_slider.layout.display = 'none' if distribution2 != 'Binomial' else 'inline-flex'
        self.a2_slider.layout.display = 'none' if distribution2 not in ['Beta', 'Gamma', 'Uniform'] else 'inline-flex'
        self.b2_slider.layout.display = 'none' if distribution2 not in ['Beta', 'Uniform'] else 'inline-flex'
        self.df2_slider.layout.display = 'none' if distribution2 != 'Chi-Square' else 'inline-flex'

    def display(self):
        self.distribution1_dropdown.observe(self.update_ui, 'value')
        self.distribution2_dropdown.observe(self.update_ui, 'value')
        self.update_ui()
        display(self.ui, self.out)

 
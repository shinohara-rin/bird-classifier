defmodule BirdClassifiler.MixProject do
  use Mix.Project

  def project do
    [
      app: :bird_classifier,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:tflite_elixir, "~> 0.3.3"}
    ]
  end
end

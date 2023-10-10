defmodule BirdClassifier do
  defstruct [:model, :input_tensor]

  def load_model(model_path) do
    model = TFLiteElixir.Interpreter.new!(model_path)
    %BirdClassifier{model: model}
  end

  def get_input_tensor(self) do
    [input_tensorn | _] = TFLiteElixir.Interpreter.inputs!(self.model)
    tensor = TFLiteElixir.Interpreter.tensor(self.model, input_tensorn)
    %BirdClassifier{self | input_tensor: tensor}
  end

  def normalize_input_image(self, image_path) do
    image = StbImage.read_file!(image_path)
    %TFLiteElixir.TFLiteTensor{shape: {_, h, w, _}} = self.input_tensor
    StbImage.resize(image, h, w)
  end

  def predict(self, image_path, opts \\ []) do
    self = get_input_tensor(self)
    image = normalize_input_image(self, image_path)
    TFLiteElixir.TFLiteTensor.set_data(self.input_tensor, image.data)
    TFLiteElixir.Interpreter.invoke!(self.model)
    output = TFLiteElixir.Interpreter.output_tensor!(self.model, 0)
    postprocess(output, opts)
  end

  def postprocess(softmax, opts) do
    label_file = opts[:label_file]
    top_k = opts[:top_k] || 3
    output = Nx.from_binary(softmax, :u8)
    sorted_idx = Nx.argsort(output, direction: :desc)
    topk_idx = Nx.take(sorted_idx, Nx.iota({top_k}))
    to_label(topk_idx, label_file)
  end

  def to_label(topk_idx, nil), do: topk_idx

  def to_label(topk_idx, label_file) when is_binary(label_file) do
    label = File.read!(label_file)
    labels = String.split(label, "\n")
    Enum.map(Nx.to_flat_list(topk_idx), fn idx -> Enum.at(labels, idx) end)
  end

  def to_label(_, label_file),
    do: raise("Expected label_file to be a string, but got `#{inspect(label_file)}`")
end

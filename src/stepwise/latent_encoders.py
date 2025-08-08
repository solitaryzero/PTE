import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def latent_encoder_factory(
    latent_type,
    **kwargs,
):
    if (latent_type == 'multihot'):
        int_precision, frac_precision = kwargs['int_precision'], kwargs['frac_precision']
        intermediate_dim = kwargs.get('intermediate_dim', 512)
        return MultiHotLatentEncoder(int_precision, frac_precision, intermediate_dim)
    elif (latent_type == 'number'):
        intermediate_dim = kwargs['intermediate_dim']
        return ValueLatentEncoder(intermediate_dim)
    elif (latent_type == 'fone'):
        int_precision, frac_precision = kwargs['int_precision'], kwargs['frac_precision']
        intermediate_dim = kwargs.get('intermediate_dim', 512)
        return FoNELatentEncoder(int_precision, frac_precision, intermediate_dim)
    else:
        raise NotImplementedError('Unsupported latent encoder')


class LatentEncoderBase:
    def __init__(self):
        self.latent_type = 'none'

    def encode(self, numbers):
        raise NotImplementedError
    
    def decode(self, latents, token_types):
        raise NotImplementedError
    
    def get_dummy_latent(self):
        raise NotImplementedError
    
    def get_projection_module(self, input_dimension):
        raise NotImplementedError
    
    def load_projection_module(self, module_path):
        raise NotImplementedError
    
    def latent_loss(self, logits, labels):
        raise NotImplementedError
    
    def convert_encoded_to_tensor(self, encoded):
        raise NotImplementedError

    def compute_metrics(self, predictions, labels):
        raise NotImplementedError


class MultiHotLatentEncoder(LatentEncoderBase):
    def __init__(self, int_precision, frac_precision, intermediate_dim=512):
        self.int_precision = int_precision
        self.frac_precision = frac_precision
        self.intermediate_dim = intermediate_dim
        self.latent_dimension = 10*(int_precision+frac_precision)+1 # sign bit
        self.latent_type = 'multihot'

    def encode_single(self, number):
        # n_str = str(number)
        n_str = format(number, f'.{self.frac_precision}f')
        n_str = n_str.rstrip('0').rstrip('.') if '.' in n_str else n_str

        latent_int = [0 for _ in range(10*self.int_precision)]
        latent_frac = [0 for _ in range(10*self.frac_precision)]

        if (n_str[0] == '-'):
            n_str = n_str[1:]
            sign_bit = [1]
        else:
            sign_bit = [0]

        segs = n_str.split('.')
        int_str = segs[0]
        if (len(int_str) == 0):
            int_str = '0'
        if (len(segs) == 2):
            frac_str = segs[1]
        else:
            frac_str = ''

        l = len(int_str)
        for i in range(min(self.int_precision, l)):
            digit = int(int_str[l-1-i])
            latent_int[i*10+digit] = 1
        for i in range(l, self.int_precision):
            latent_int[i*10] = 1

        l = len(frac_str)
        for i in range(min(self.frac_precision, l)):
            digit = int(frac_str[i])
            latent_frac[i*10+digit] = 1
        for i in range(l, self.frac_precision):
            latent_frac[i*10] = 1

        full_latent = sign_bit + latent_int + latent_frac
        return full_latent
    
    def encode(self, numbers):
        latents = []
        for x in numbers:
            latents.append(self.encode_single(x))

        return latents

    def decode(self, latents, token_types):
        results = []
        for index, latent in enumerate(latents):
            lat = latent[-1]
            if (token_types[index][-1] == 0):
                results.append(None)
                continue

            if (lat[0] == 1):
                sign_bit = '-'
            else:
                sign_bit = ''

            int_str = ''
            for i in range(self.int_precision*10):
                if (lat[1+i] == 1):
                    int_str += str(i%10)
            int_str = int_str.rstrip('0')

            frac_str = ''
            for i in range(self.frac_precision*10):
                if (lat[i+1+self.int_precision*10] == 1):
                    frac_str += str(i%10)
            frac_str = frac_str.rstrip('0')

            if (int_str == ''):
                int_str = '0'
            else:
                int_str = int_str[::-1]

            if (frac_str == ''):
                results.append(f'{sign_bit}{int_str}')
            else:
                results.append(f'{sign_bit}{int_str}.{frac_str}')

        return results
    
    def get_dummy_latent(self):
        return [0 for _ in range(self.latent_dimension)]
    
    def get_projection_module(self, input_dimension):
        # projector = nn.Linear(input_dimension, self.latent_dimension)
        projector = nn.Sequential(
            nn.Linear(input_dimension, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.latent_dimension)
        )
        return projector
    
    def load_projection_module(self, module_path):
        return super().load_projection_module(module_path)
    
    def latent_loss(self, logits, labels):
        logits = logits.to(torch.float32)
        labels = labels.to(logits.device)
        latent_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        latent_loss = torch.mean(latent_loss, dim=-1)
        return latent_loss
    
    def convert_encoded_to_tensor(self, encoded):
        return torch.tensor(encoded, dtype=torch.float32)
    
    @torch.no_grad()
    def compute_metrics(self, predictions, labels, probe_mask=None):
        comparison = ((torch.sigmoid(predictions) >= 0.5) == labels)
        comparison = torch.all(comparison, dim=-1)
        if (probe_mask is not None):
            comparison = comparison[probe_mask].flatten()
        else:
            comparison = comparison.flatten()

        return {
            'total': comparison.shape[0],
            'correct': comparison.sum().item(),
        }
    

class ValueLatentProjector(nn.Module):
    def __init__(self, input_dim, intermediate_dim=512):
        super(ValueLatentProjector, self).__init__()
        # self.sign_projector = nn.Linear(input_dim, 1)
        # self.value_regressor = nn.Linear(input_dim, 1)

        self.sign_projector = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1)
        )
        self.value_regressor = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1)
        )

    def forward(self, inputs):
        sign = self.sign_projector(inputs)
        value = self.value_regressor(inputs)
        return torch.cat([sign, value], dim=-1)


class ValueLatentEncoder(LatentEncoderBase):
    def __init__(self, intermediate_dim):
        self.latent_type = 'number'
        self.intermediate_dim = intermediate_dim
        self.eps = 1e-5

    def encode_single(self, number):
        if (number < 0):
            sign_bit = 1
            number = abs(number)
        else:
            sign_bit = 0

        if (number == 0):
            x = self.eps
        else:
            x = number
        return [sign_bit, np.log2(x)]
        # return [sign_bit, number]
    
        # return [number]
    
    def encode(self, numbers):
        latents = []
        for x in numbers:
            latents.append(self.encode_single(x))

        return latents

    def decode(self, latents, token_types):
        results = []
        for index, latent in enumerate(latents):
            lat = latent[-1]
            if (token_types[index][-1] == 0):
                results.append(None)
                continue

            # results.append(str(np.exp2(lat)))
            # results.append(str(lat))

            sign = '-' if (torch.sigmoid(lat[0]) >= 0.5) else ''
            value = str(np.exp2(lat[1]))
            # value = str(lat[1])
            results.append(sign+value) 

        return results
    
    def get_dummy_latent(self):
        return [0, 0]
        # return [0]
    
    def get_projection_module(self, input_dimension=-1):
        # projector = nn.Sequential(
        #     nn.Linear(input_dimension, self.intermediate_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.intermediate_dim, 2)
        # )
        projector = ValueLatentProjector(input_dim=input_dimension, intermediate_dim=self.intermediate_dim)
        # projector = nn.Linear(input_dimension, 1)
        return projector
    
    def load_projection_module(self, module_path):
        return super().load_projection_module(module_path)
    
    def latent_loss(self, logits, labels):
        # loss_fct = nn.MSELoss(reduction='none')
        # labels = labels.to(logits.device).to(logits.dtype)
        # latent_loss = loss_fct(logits, labels)

        # return latent_loss
        
        logits = logits.to(torch.float32)
        labels = labels.to(torch.float32)

        sign_logits = logits[..., 0]
        sign_labels = labels[..., 0]
        sign_loss = F.binary_cross_entropy_with_logits(sign_logits, sign_labels, reduction='none')

        value_logits = logits[..., 1]
        value_labels = labels[..., 1]
        value_loss = F.mse_loss(value_logits, value_labels, reduction='none')

        return sign_loss + value_loss
    
    def convert_encoded_to_tensor(self, encoded):
        return torch.tensor(encoded, dtype=torch.float32)
    
    @torch.no_grad()
    def compute_metrics(self, predictions, labels, probe_mask=None):
        sign_correctness = ((torch.sigmoid(predictions[..., 0]) >= 0.5) == labels[..., 0])
        predicted_value, golden_value = torch.exp2(predictions[..., 1]), torch.exp2(labels[..., 1])
        delta = torch.abs(predicted_value - golden_value)
        threshold = golden_value * 0.01
        comparison = (delta < threshold) & sign_correctness
        if (probe_mask is not None):
            comparison = comparison[probe_mask].flatten()
        else:
            comparison = comparison.flatten()

        # print(predicted_value[probe_mask])
        # print(golden_value[probe_mask])
        # input()

        return {
            'total': comparison.shape[0],
            'correct': comparison.sum().item(),
        }
    

class FoNELatentProjector(nn.Module):
    def __init__(self, input_dim, value_dimension):
        super(FoNELatentProjector, self).__init__()
        self.sign_projector = nn.Linear(input_dim, 1)
        self.value_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, value_dimension),
        )

    def forward(self, inputs):
        sign = self.sign_projector(inputs)
        value = self.value_regressor(inputs)
        return torch.cat([sign, value], dim=-1)


class FoNELatentEncoder(LatentEncoderBase):
    def __init__(self, int_precision, frac_precision, intermediate_dim=512):
        self.int_precision = int_precision
        self.frac_precision = frac_precision
        self.intermediate_dim = intermediate_dim
        self.value_dimension = 2*(int_precision+frac_precision)
        self.latent_dimension = self.value_dimension+1 # sign bit
        self.latent_type = 'fone'

        self._init_bases()

    def _init_bases(self):
        sine_bases = []
        cosine_bases = []
        base = 2 * np.pi / 10
        for x in range(10):
            sine_bases.append(np.sin(x * base))
            cosine_bases.append(np.cos(x * base))

        self.sin_bases = sine_bases
        self.cos_bases = cosine_bases

        self.sin_cos_map = torch.tensor([cosine_bases, sine_bases])

        ten_bases = []
        for i in range(self.int_precision):
            ten_bases.append(np.power(10.0, i+1))

        for i in range(self.frac_precision):
            ten_bases.append(np.power(10.0, -i))
        self.ten_bases = ten_bases

    def encode_single(self, number):
        n_str = format(number, f'.{self.frac_precision}f')
        n_str = n_str.rstrip('0').rstrip('.') if '.' in n_str else n_str

        latent_int = [0 for _ in range(2*self.int_precision)]
        latent_frac = [0 for _ in range(2*self.frac_precision)]

        if (n_str[0] == '-'):
            n_str = n_str[1:]
            number = abs(number)
            sign_bit = [1]
        else:
            sign_bit = [0]

        for i in range(self.int_precision):
            base = number / self.ten_bases[i]
            latent_int[i*2] = np.cos(2*np.pi*base)
            latent_int[i*2+1] = np.sin(2*np.pi*base)

        frac_number = number - int(number)
        for i in range(self.frac_precision):
            base = frac_number / self.ten_bases[i]
            latent_frac[i*2] = np.cos(2*np.pi*base)
            latent_frac[i*2+1] = np.sin(2*np.pi*base)

        full_latent = sign_bit + latent_int + latent_frac
        return full_latent
    
    def encode(self, numbers):
        latents = []
        for x in numbers:
            latents.append(self.encode_single(x))

        return latents

    def decode(self, latents, token_types, eps=1e-5):
        results = []
        for index, latent in enumerate(latents):
            lat = latent[-1] # decode only last token
            if (token_types[index][-1] == 0):
                results.append(None)
                continue

            if (lat[0] == 1):
                sign_bit = '-'
            else:
                sign_bit = ''

            value_tensor = lat[1:].reshape(-1, 2)
            int_str = ''
            for i in range(self.int_precision):
                vec = value_tensor[i]
                angle = torch.atan2(vec[1], vec[0])
                digit = angle / (torch.pi * 2) * 10
                digit = torch.floor(digit + eps).item()
                int_str += str(int(digit))
            int_str = int_str.rstrip('0')

            frac_str = ''
            for i in range(self.frac_precision):
                vec = value_tensor[i + self.int_precision]
                angle = torch.atan2(vec[1], vec[0])
                digit = angle / (torch.pi * 2) * 10
                digit = torch.floor(digit + eps).item()
                frac_str += str(int(digit))
            frac_str = frac_str.rstrip('0')

            int_str = int_str[::-1]

            if (frac_str == ''):
                results.append(f'{sign_bit}{int_str}')
            else:
                results.append(f'{sign_bit}{int_str}.{frac_str}')

        return results
    
    def get_dummy_latent(self):
        return [0 for _ in range(self.latent_dimension)]
    
    def get_projection_module(self, input_dimension):
        projector = FoNELatentProjector(input_dim=input_dimension, value_dimension=self.value_dimension)
        return projector
    
    def load_projection_module(self, module_path):
        return super().load_projection_module(module_path)
    
    def latent_loss(self, logits, labels):
        logits = logits.to(torch.float32).to(labels.device)
        labels = labels.to(torch.float32)

        sign_logits = logits[..., 0]
        sign_labels = labels[..., 0]
        sign_loss = F.binary_cross_entropy_with_logits(sign_logits, sign_labels, reduction='none')

        k = (logits.shape[-1]-1) // 2
        new_shape = logits.shape[:-1] + (k, 2)
        value_logits = logits[..., 1:].view(new_shape)
        value_labels = labels[..., 1:].view(new_shape)
        value_loss = 1 - F.cosine_similarity(value_logits, value_labels, dim=-1)
        value_loss = torch.mean(value_loss, dim=-1)

        return sign_loss + value_loss
    
    def convert_encoded_to_tensor(self, encoded):
        return torch.tensor(encoded, dtype=torch.float32)
    
    @torch.no_grad()
    def compute_metrics(self, predictions, labels, probe_mask=None):
        sign_correctness = ((torch.sigmoid(predictions[..., 0]) >= 0.5) == labels[..., 0])

        k = (predictions.shape[-1]-1) // 2
        new_shape = predictions.shape[:-1] + (k, 2)
        sin_cos_map = self.sin_cos_map.to(labels.dtype).to(labels.device)

        prediction_values = torch.reshape(predictions[..., 1:], new_shape) # [bsz, len, k, 2]
        prediction_sim = prediction_values @ sin_cos_map # [bsz, len, k, 10]
        labels_values = torch.reshape(labels[..., 1:], new_shape) # [bsz, len, k, 2]
        labels_sim = labels_values @ sin_cos_map # [bsz, len, k, 10]
        # print(prediction_values[probe_mask])
        # print(torch.argmax(prediction_sim, dim=-1)[probe_mask])
        # print(torch.argmax(labels_sim, dim=-1)[probe_mask])
        # input()
        value_correctness = (torch.argmax(prediction_sim, dim=-1) == torch.argmax(labels_sim, dim=-1))
        value_correctness = torch.all(value_correctness, dim=-1)

        comparison = sign_correctness & value_correctness
        if (probe_mask is not None):
            comparison = comparison[probe_mask].flatten()
        else:
            comparison = comparison.flatten()

        return {
            'total': comparison.shape[0],
            'correct': comparison.sum().item(),
        }
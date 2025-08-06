import torch

class MultiHotLatentEncoder:
    def __init__(self, int_precision, frac_precision):
        self.int_precision = int_precision
        self.frac_precision = frac_precision


    def encode(self, number):
        n_str = str(number)
        latent_int = [0 for _ in range(10*self.int_precision)]
        latent_frac = [0 for _ in range(10*self.frac_precision)]

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

        full_latent = latent_int + latent_frac
        return full_latent

    def get_dummy_latent(self):
        return [0 for _ in range(10*(self.int_precision+self.frac_precision))]


    def decode(self, latents, token_types):
        results = []
        for index, latent in enumerate(latents):
            lat = latent[-1]
            if (token_types[index][-1] == 0):
                results.append(None)
                continue

            int_str = ''
            for i in range(self.int_precision*10):
                if (lat[i] == 1):
                    int_str += str(i%10)
            int_str = int_str.rstrip('0')

            frac_str = ''
            for i in range(self.frac_precision*10):
                if (lat[i+self.int_precision*10] == 1):
                    frac_str += str(i%10)
            frac_str = frac_str.rstrip('0')

            if (int_str == ''):
                int_str = '0'
            else:
                int_str = int_str[::-1]

            if (frac_str == ''):
                results.append(int_str)
            else:
                results.append(f'{int_str}.{frac_str}')

        return results
    

class FoneLatentEncoder:
    def __init__(self, int_precision, frac_precision):
        self.int_precision = int_precision
        self.frac_precision = frac_precision


    def _create_precomputed_cos_sin_matrix(self, period_base_list=[2, 5]):
        """
        Creates a precomputed cos/sin matrix for the given period base list and number of positions.
        """
        # Convert string periods to floats if necessary
        period_base_list = [base if type(base) != str else eval(base) for base in period_base_list]
        num_positions = 10  # Modify as needed for desired number of positions
        base_positions = torch.arange(num_positions)
        cos_sin_list = []
        # Compute cos and sin values for each period
        for period in period_base_list:
            w = 2 * torch.pi / period
            cos_sin_list.append(torch.cos(w * base_positions))
            cos_sin_list.append(torch.sin(w * base_positions))
        
        # Stack all values to form a matrix
        cos_sin_matrix = torch.stack(cos_sin_list, dim=1)
        return cos_sin_matrix


    def encode(self, number):
        n_str = str(number)
        latent_int = [0 for _ in range(10*self.int_precision)]
        latent_frac = [0 for _ in range(10*self.frac_precision)]

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

        full_latent = latent_int + latent_frac
        return full_latent

    def get_dummy_latent(self):
        return [0 for _ in range(10*(self.int_precision+self.frac_precision))]


    def decode(self, latents, token_types):
        results = []
        for index, latent in enumerate(latents):
            lat = latent[-1]
            if (token_types[index][-1] == 0):
                results.append(None)
                continue

            int_str = ''
            for i in range(self.int_precision*10):
                if (lat[i] == 1):
                    int_str += str(i%10)
            int_str = int_str.rstrip('0')

            frac_str = ''
            for i in range(self.frac_precision*10):
                if (lat[i+self.int_precision*10] == 1):
                    frac_str += str(i%10)
            frac_str = frac_str.rstrip('0')

            if (int_str == ''):
                int_str = '0'
            else:
                int_str = int_str[::-1]

            if (frac_str == ''):
                results.append(int_str)
            else:
                results.append(f'{int_str}.{frac_str}')

        return results
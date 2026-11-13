import jwt, { JwtPayload } from "jsonwebtoken";

export interface AppJwtPayload extends JwtPayload {
  userId: string;
  email: string;
  role: string;
}

export const createToken = (payload: AppJwtPayload) => {
  return jwt.sign(payload, process.env.JWT_SECRET!, {
    expiresIn: "7d",
  });
};

export const verifyToken = (token: string): AppJwtPayload => {
  return jwt.verify(token, process.env.JWT_SECRET!) as AppJwtPayload;
};
